#Libraries
import argparse
import gc
import json
import logging
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

from config import COMMON_FEATURES, MODELS_DIR, RESULTS_DIR
from loaders import DATASETS, available_datasets
from preprocessing import build_xy, split
from progress import Progress

#Operating points to report
FPR_TARGETS = (0.05, 0.01, 0.001, 0.0001)
#Attack prevalence
PREVALENCES = (0.5, 0.01, 0.001, 0.0001)
FLOWS_PER_DAY = 10_000_000 #reference network size for the alert-volume column

log = logging.getLogger("oppoints")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "operating_points.log", encoding="utf-8")])


'''Full binary test split for a dataset'''
def test_split_for(name):
    if name == "combined":
        cache = RESULTS_DIR / "common_cache"
        frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
        if not frames:
            raise FileNotFoundError("common_cache is empty")
        df = pd.concat(frames, ignore_index=True)
        X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
        y, y_bin = df["label"].astype(str), df["binary"].astype(int)
    else:
        loader, _ = DATASETS[name]
        X, y, y_bin = build_xy(loader())
    _, X_te, _, _, _, yb_te = split(X, y, y_bin, dataset=name)
    return X_te, yb_te.map({0: "Benign", 1: "Attack"}).astype(str)


'''Positive predictive value at a given attack prevalence (Bayes, not test-set ratio)'''
def ppv(tpr, fpr, prevalence):
    denom = tpr * prevalence + fpr * (1.0 - prevalence)
    return float(tpr * prevalence / denom) if denom > 0 else 0.0


'''Rows of the operating-point table for one model'''
def points_for(y_true, score, dataset, model, n_neg):
    fpr, tpr, thr = roc_curve((np.asarray(y_true) == "Attack").astype(int), score)
    resolution = 1.0 / n_neg if n_neg else None #finest FPR the test split can express at all
    rows = []
    for target in FPR_TARGETS:
        ok = np.where(fpr <= target)[0]
        i = ok[-1] if len(ok) else 0
        t, f = float(tpr[i]), float(fpr[i])
        #FPR of exactly 0 means "below the resolution of this test split"
        f_eff = max(f, resolution) if resolution else f
        rec = {"dataset": dataset, "model": model, "fpr_target": target,
               "tpr": t, "actual_fpr": f, "fpr_used_for_projection": f_eff,
               "threshold": float(thr[i]) if np.isfinite(thr[i]) else None,
               "n_negatives": int(n_neg), "fpr_resolution": resolution,
               #False when the FPR is finer than a single benign row
               "measurable": bool(resolution is not None and target >= resolution),
               "false_alerts_per_day": float(f_eff * FLOWS_PER_DAY)}
        for pi in PREVALENCES:
            rec[f"ppv_at_prev_{pi}"] = ppv(t, f_eff, pi)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Alert-budget operating points and base-rate corrected precision.")
    ap.add_argument("datasets", nargs="*")
    args = ap.parse_args()
    wanted = args.datasets or (list(available_datasets()) + ["combined"])
    prog = Progress(max(len(wanted), 1), phase="operating_points")
    out = []
    for name in wanted:
        if name != "combined" and name not in DATASETS:
            prog.advance()
            continue
        prog.update(stage="loading", dataset=name, message=f"full test split for {name}")
        try:
            X_te, y_te = test_split_for(name)
        except Exception as e:
            log.warning("skip %s: %s", name, e)
            prog.advance()
            continue
        n_neg = int((y_te == "Benign").sum())
        log.info("%s: %d test rows, %d benign -> finest measurable FPR = %.2e", name, len(y_te), n_neg, (1.0 / n_neg) if n_neg else float("nan"))
        if n_neg == 0:
            log.warning("%s has no benign rows in test, skipping", name)
            prog.advance()
            continue
        models = sorted(MODELS_DIR.glob(f"{name}__binary__*.joblib"))
        for p in models:
            model = p.stem.split("__", 2)[2]
            prog.update(stage="scoring", dataset=name, model=model, message=f"{name}/{model}")
            try:
                pipe = joblib.load(p)
                pr = pipe.predict_proba(X_te)
                cls = list(getattr(pipe, "classes_", ["Attack", "Benign"]))
                score = pr[:, cls.index("Attack")] #probability of the positive class
                out += points_for(y_te, score, name, model, n_neg)
                del pipe, pr
            except Exception as e:
                log.warning("  skip %s/%s: %s", name, model, e) #models without predict_proba
            gc.collect()
        del X_te, y_te
        gc.collect()
        prog.advance()

    if not out:
        log.warning("no operating points produced")
        prog.finish("operating_points: nothing to write")
        return
    d = pd.DataFrame(out)
    d.to_csv(RESULTS_DIR / "table_operating_points.csv", index=False)
    (RESULTS_DIR / "operating_points.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    #headline view: best model per dataset at the tightest measurable operating point
    m = d[d.measurable]
    if not m.empty:
        best = m.loc[m.groupby(["dataset", "fpr_target"])["tpr"].idxmax()].reset_index(drop=True)
        log.info("\nTightest measurable operating points (PPV at 0.1%% attack prevalence):\n%s",
                 best[["dataset", "fpr_target", "model", "tpr", "actual_fpr", "ppv_at_prev_0.001", "false_alerts_per_day"]].round(4).to_string(index=False))
    prog.finish(f"operating_points: {len(out)} rows")
    log.info("Completed: %d operating points over %d models", len(out), d.model.nunique())


if __name__ == "__main__":
    main()
