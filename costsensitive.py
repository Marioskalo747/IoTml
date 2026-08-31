#Libraries
import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

from config import FN_COST_RATIO, RESULTS_DIR, SLOW_MODEL_TRAIN_CAP
from evaluation import evaluate_model
from loaders import DATASETS, available_datasets
from models_zoo import LabelEncodedClassifier, model_zoo
from preprocessing import build_xy, make_preprocessor, split, split_meta
from progress import Progress

#  none: the baseline already reported elsewhere
#  balanced: every class weighted by inverse frequency
#  fn_cost: balanced weights, then attack classes multiplied by FN_COST_RATIO
STRATEGIES = ("none", "balanced", "fn_cost")
DEFAULT_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DEFAULT_COSTS = (1.0, 5.0, 10.0) #how many false alarms one missed attack is worth

log = logging.getLogger("cost")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "cost_sensitive.log", encoding="utf-8")])


'''Per-row training weights for a strategy'''
def sample_weights(y, strategy, fn_cost):
    if strategy == "none":
        return None
    w = compute_sample_weight("balanced", y) #inverse frequency, the usual starting point
    if strategy == "fn_cost":
        #an attack row costs fn_cost times a benign row: missing it is that much worse
        w = w * np.where(np.asarray(y).astype(str) == "Benign", 1.0, float(fn_cost))
    return w


'''Fit one model with optional sample weights, handling the LabelEncoded wrapper'''
def fit_weighted(spec, X_tr, y_tr, w):
    pipe = Pipeline([("pre", make_preprocessor(X_tr)), ("clf", spec.build())])
    t0 = time.perf_counter()
    if w is None:
        pipe.fit(X_tr, y_tr.astype(str))
        return pipe, time.perf_counter() - t0, True
    try:
        pipe.fit(X_tr, y_tr.astype(str), clf__sample_weight=w)
        return pipe, time.perf_counter() - t0, True
    except (TypeError, ValueError) as e:
        log.warning("  %s does not accept sample_weight (%s), training unweighted", spec.name, e)
        pipe = Pipeline([("pre", make_preprocessor(X_tr)), ("clf", spec.build())])
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr.astype(str))
        return pipe, time.perf_counter() - t0, False


def main():
    ap = argparse.ArgumentParser(description="Cost-sensitive training: does weighting the rare attack classes help?")
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--costs", nargs="*", type=float, default=list(DEFAULT_COSTS))
    ap.add_argument("--task", choices=["multiclass", "binary", "both"], default="multiclass")
    args = ap.parse_args()
    wanted = [d for d in (args.datasets or available_datasets()) if d in DATASETS]
    tasks = ["multiclass", "binary"] if args.task == "both" else [args.task]
    costs = sorted({1.0} | {c for c in args.costs if c > 0}) #cost 1.0 reproduces "balanced"
    rows = []
    #carry forward whatever this invocation is not recomputing
    prev_path = RESULTS_DIR / "cost_sensitive.json"
    if prev_path.exists():
        try:
            rows = [r for r in json.loads(prev_path.read_text(encoding="utf-8"))
                    if r.get("dataset") not in wanted]
            if rows:
                log.info("keeping %d rows from the previous run", len(rows))
        except Exception as e:
            log.warning("could not reuse %s (%s); it will be rebuilt from this run only", prev_path.name, e)
    prog = Progress(max(len(wanted), 1), phase="cost_sensitive")

    for name in wanted:
        prog.update(stage="loading", dataset=name, message=f"loading {name}")
        try:
            loader, _ = DATASETS[name]
            X, y, y_bin = build_xy(loader())
        except Exception as e:
            log.warning("skip %s: %s", name, e)
            prog.advance()
            continue
        X_tr, X_te, y_tr, y_te, yb_tr, yb_te = split(X, y, y_bin, dataset=name)
        del X, y, y_bin
        gc.collect()
        for task in tasks:
            a_tr, a_te = ((y_tr, y_te) if task == "multiclass"
                          else (yb_tr.map({0: "Benign", 1: "Attack"}), yb_te.map({0: "Benign", 1: "Attack"})))
            labels = sorted(pd.unique(pd.concat([a_tr, a_te]).astype(str)))
            for mname in args.models:
                spec = next((s for s in model_zoo(len(labels)) if s.name == mname), None)
                if spec is None:
                    continue #model not installed
                Xt, yt = (X_tr, a_tr)
                if spec.slow and len(X_tr) > SLOW_MODEL_TRAIN_CAP:
                    idx = (pd.Series(range(len(X_tr)), index=X_tr.index).groupby(a_tr, group_keys=False)
                           .apply(lambda s: s.sample(n=max(1, round(len(s) * SLOW_MODEL_TRAIN_CAP / len(X_tr))), random_state=42)))
                    Xt, yt = X_tr.loc[idx.index], a_tr.loc[idx.index]
                for strategy in STRATEGIES:
                    #"fn_cost" is swept over the cost list and the other two have a single setting
                    for fn_cost in (costs if strategy == "fn_cost" else [1.0]):
                        prog.update(stage="training", dataset=name, task=task, model=mname,
                                    substage=f"{strategy} cost={fn_cost:g}",
                                    message=f"{name}/{task}/{mname}/{strategy}")
                        try:
                            w = sample_weights(yt, strategy, fn_cost)
                            pipe, dt, weighted = fit_weighted(spec, Xt, yt, w)
                            proba = None
                            try:
                                proba = pipe.predict_proba(X_te)
                            except Exception:
                                pass
                            r = evaluate_model(pipe, X_te, a_te.astype(str), labels, proba=proba)
                            rec = {"dataset": name, "task": task, "model": mname,
                                   "strategy": strategy, "fn_cost": fn_cost,
                                   **split_meta(name, a_tr, a_te),
                                   "weights_applied": weighted, "train_rows": len(Xt), "train_time_s": dt,
                                   **{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro",
                                                        "f1_weighted", "precision_macro", "recall_macro",
                                                        "mcc", "g_mean")}}
                            #per-class recall f1_macro can't move while a rare class doubles recall
                            for pc in r["per_class"]:
                                rec[f"recall__{pc['class']}"] = pc["recall"]
                                rec[f"f1__{pc['class']}"] = pc["f1"]
                            rows.append(rec)
                            log.info("  %s/%s %-13s %-9s cost=%-4g f1M=%.4f bal=%.4f recall_macro=%.4f",
                                     name, task, mname, strategy, fn_cost, rec["f1_macro"],
                                     rec["balanced_accuracy"], rec["recall_macro"])
                            del pipe
                        except Exception as e:
                            log.warning("  FAIL %s/%s/%s/%s: %s", name, task, mname, strategy, e)
                        gc.collect()
                        if rows:
                            pd.DataFrame(rows).to_csv(RESULTS_DIR / "cost_sensitive_partial.csv", index=False)
        del X_tr, X_te, y_tr, y_te, yb_tr, yb_te
        gc.collect()
        prog.advance()

    if not rows:
        log.warning("no cost-sensitive results")
        prog.finish("cost_sensitive: nothing to write")
        return
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_DIR / "table_cost_sensitive.csv", index=False)
    (RESULTS_DIR / "cost_sensitive.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    (RESULTS_DIR / "cost_sensitive_partial.csv").unlink(missing_ok=True)
    #what changed against the unweighted baseline
    base = d[d.strategy == "none"].set_index(["dataset", "task", "model"])
    gains = []
    for _, r in d[d.strategy != "none"].iterrows():
        k = (r["dataset"], r["task"], r["model"])
        if k in base.index:
            b = base.loc[k]
            gains.append({"dataset": r["dataset"], "task": r["task"], "model": r["model"],
                          "strategy": r["strategy"], "fn_cost": r["fn_cost"],
                          "d_f1_macro": r["f1_macro"] - b["f1_macro"],
                          "d_balanced_accuracy": r["balanced_accuracy"] - b["balanced_accuracy"],
                          "d_recall_macro": r["recall_macro"] - b["recall_macro"]})
    if gains:
        g = pd.DataFrame(gains)
        g.to_csv(RESULTS_DIR / "table_cost_sensitive_gains.csv", index=False)
        log.info("\nChange versus the unweighted baseline (mean over models):\n%s",
                 g.groupby(["dataset", "task", "strategy", "fn_cost"])[
                     ["d_f1_macro", "d_balanced_accuracy", "d_recall_macro"]].mean().round(4).to_string())
    prog.finish(f"cost_sensitive: {len(rows)} experiments")
    log.info("Completed: %d experiments", len(rows))


if __name__ == "__main__":
    main()
