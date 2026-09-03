#Libraries
import argparse
import gc
import json
import logging
import re #regex for port columns
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMON_FEATURES, RESULTS_DIR
from evaluation import evaluate_model
from loaders import DATASETS, available_datasets
from models_zoo import model_zoo
from preprocessing import (PER_IP_AGG, build_xy, make_preprocessor,
                           port_columns as _port_columns, split, split_meta)
from progress import Progress

#the strongest models
DEFAULT_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]

#log
log = logging.getLogger("leak")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "leakage_ablation.log", encoding="utf-8")])


'''find port columns (shared definition, aggregates excluded)'''
def port_columns(cols):
    return _port_columns(cols)

'''find per-IP aggregate columns'''
def agg_columns(cols):
    return [c for c in cols if c in PER_IP_AGG]

'''find the windowed context columns added by loaders.add_window_features'''
def window_columns(cols):
    return [c for c in cols if str(c).startswith(("w5_", "w10_"))]

#What removing a group can tell you
GROUP_ROLE = {"ports": "shortcut_risk", "ip_aggregates": "shortcut_risk", "window_context": "deployable_signal"}

'''prepare train-test split for a dataset'''
def load_split(name):
    if name == "combined": #combined dataset from common features cache
        cache = RESULTS_DIR / "common_cache"
        frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
        df = pd.concat(frames, ignore_index=True)
        X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
        y = df["label"].astype(str)
        y_bin = df["binary"].astype(int)
    else:
        loader, _ = DATASETS[name]
        X, y, y_bin = build_xy(loader()) #datasets full feature set
    return split(X, y, y_bin, dataset=name)  #same split (dedup + stratify + cap)

'''preprocessing and model pipeline (returns metrics and training time)'''
def fit_score(spec, X_tr, y_tr, X_te, y_te, labels):
    #rebuilt percondition 
    pipe = Pipeline([("pre", make_preprocessor(X_tr)), ("clf", spec.build())])
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr.astype(str))
    dt = time.perf_counter() - t0
    proba = None
    try:
        proba = pipe.predict_proba(X_te)
    except Exception:
        pass #model without predict_proba or failed to compute probabilities
    res = evaluate_model(pipe, X_te, y_te.astype(str), labels, proba=proba)
    del pipe
    gc.collect()
    return res, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()
    wanted = args.datasets or (list(available_datasets()) + ["combined"])
    out = []
    prev_path = RESULTS_DIR / "leakage_ablation.json"
    if prev_path.exists():
        try:
            out = [r for r in json.loads(prev_path.read_text(encoding="utf-8")) if r.get("dataset") not in wanted]
            if out:
                log.info("keeping %d rows from the previous run", len(out))
        except Exception as e:
            log.warning("could not reuse %s (%s); it will be rebuilt from this run only", prev_path.name, e)
    prog = Progress(len(wanted), phase="leakage_ablation")
    for name in wanted:
        prog.update(stage="ablation", dataset=name, message=f"leakage ablation {name}")
        if name != "combined" and name not in DATASETS:
            continue #unknown dataset
        X_tr, X_te, y_tr, y_te, yb_tr, yb_te = load_split(name)
        pc = port_columns(X_tr.columns)
        ac = agg_columns(X_tr.columns)
        wc = window_columns(X_tr.columns)
        conditions = {"full": []}
        if pc:
            conditions["no_ports"] = pc
        if ac:
            conditions["no_ip_aggregates"] = ac
        if wc:
            conditions["no_window_context"] = wc
        if pc or ac:
            conditions["deployable"] = pc + ac
        if len(conditions) == 1:
            log.info("%s: no port, aggregate or windowed columns to ablate", name)
            del X_tr, X_te
            gc.collect()
            continue

        log.info("%s (%d features): ports=%d, IP aggregates=%d, windowed=%d", name, X_tr.shape[1], len(pc), len(ac), len(wc))
        #every combination of model, task, and condition is a separate experiment
        for task, a_tr, a_te in (("multiclass", y_tr, y_te),("binary", yb_tr.map({0: "Benign", 1: "Attack"}),yb_te.map({0: "Benign", 1: "Attack"}))):
            labels = sorted(pd.unique(pd.concat([a_tr, a_te]).astype(str)))
            for mname in args.models:
                spec = next((s for s in model_zoo(len(labels)) if s.name == mname), None)
                if spec is None:
                    continue #not installed model
                base = None #full F1 score for comparison
                for cond, drop in conditions.items():
                    try:
                        Xa = X_tr.drop(columns=drop) if drop else X_tr
                        Xb = X_te.drop(columns=drop) if drop else X_te
                        r, dt = fit_score(spec, Xa, a_tr, Xb, a_te, labels)
                        #which families were actually removed
                        kinds = "+".join(k for k, present in (("ports", any(c in pc for c in drop)), ("ip_aggregates", any(c in ac for c in drop)),
                                                              ("window_context", any(c in wc for c in drop))) if present) or "none"
                        roles = "+".join(sorted({GROUP_ROLE[k] for k in kinds.split("+") if k in GROUP_ROLE})) or "none"
                        rec = {"dataset": name, "task": task, "model": mname,"condition": cond,
                               **split_meta(name, a_tr, a_te), "n_dropped": len(set(drop)),"n_features": int(Xa.shape[1]), "train_time_s": dt,
                               "dropped_kinds": kinds, "group_role": roles, "same_as_no_ports": bool(cond == "deployable" and not ac),
                               **{k: r[k] for k in ("accuracy", "balanced_accuracy","f1_macro", "mcc", "g_mean")}}
                        if cond == "full":
                            base = rec["f1_macro"] #record the baseline before ablation
                            rec["f1_macro_drop_vs_full"] = 0.0
                        else:
                            #how much the F1 score dropped compared to the full model
                            rec["f1_macro_drop_vs_full"] = (base - rec["f1_macro"] if base is not None else None)
                        out.append(rec)
                        log.info("  %s/%s %-13s %-17s f1=%.4f  (πτώση %+.4f)",name, task, mname, cond, rec["f1_macro"],-(rec["f1_macro_drop_vs_full"] or 0))
                    except Exception as e:
                        log.warning("  FAIL %s/%s %s %s: %s", name, task, mname, cond, e)
                    gc.collect()
        del X_tr, X_te
        gc.collect()
        prog.advance()
        
    #persist results to json and csv for further analysis
    (RESULTS_DIR / "leakage_ablation.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    if out:
        d = pd.DataFrame(out)
        d.to_csv(RESULTS_DIR / "table_leakage_ablation.csv", index=False) #mean F1 per dataset, task, condition (deployable gap)
        piv = d.pivot_table(index=["dataset", "task"], columns="condition",values="f1_macro", aggfunc="mean")
        log.info("\nAvg f1_macro by condition:\n%s", piv.round(4).to_string())
    prog.finish(f"leakage_ablation: {len(out)} experiments")
    log.info("Completed: %d experiments", len(out))


if __name__ == "__main__":
    main()

