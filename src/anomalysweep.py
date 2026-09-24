#Libraries
import gc
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMON_FEATURES, RANDOM_STATE, RESULTS_DIR
from loaders import available_datasets
from progress import Progress


BENIGN_CAP = 60000  #max benign training rows
FPR_TARGETS = (0.001, 0.01, 0.05)  #operating points false alarms
CONTAMINATIONS = (0.005, 0.01, 0.02, 0.05, 0.10) #traffic flagged as attack (for thresholding)

#log 
log = logging.getLogger("anomsweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(RESULTS_DIR / "anomaly_sweep.log", encoding="utf-8")])

'''loads combined dataset from cache and cleans features'''
def load_common():
    cache = RESULTS_DIR / "common_cache"
    frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
    df = pd.concat(frames, ignore_index=True)
    df[COMMON_FEATURES] = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    #understate the false-positive rate
    n_raw = len(df)
    df = df[~df.duplicated(subset=COMMON_FEATURES + ["label"])].reset_index(drop=True)
    log.info("dedup: %d -> %d rows (%.1f%% exact duplicates dropped)", n_raw, len(df), 100 * (1 - len(df) / max(n_raw, 1)))
    return df

'''fits IsolationForest on benignand returns scores of test set'''
def fit_scores(benign_train, test_df):
    sc = StandardScaler().fit(benign_train[COMMON_FEATURES]) #scaler learned on benign only
    clf = IsolationForest(n_estimators=200, contamination="auto",random_state=RANDOM_STATE, n_jobs=-1).fit(sc.transform(benign_train[COMMON_FEATURES]))
    #higher scores = more anomalous 1=attack, 0=benign
    s = -clf.score_samples(sc.transform(test_df[COMMON_FEATURES]))
    y = (test_df["binary"] == 1).astype(int).values
    return s, y, sc, clf

'''raw scores to metrics (ROC-AUC, TPR, FPR)'''
def metrics_from_scores(s, y):
    out = {"roc_auc": float(roc_auc_score(y, s)),"n_test": int(len(y)), "attack_share": float(y.mean())}
    fpr, tpr, thr = roc_curve(y, s)
    for t in FPR_TARGETS:
        ok = np.where(fpr <= t)[0] #roc points with FPR below target
        i = ok[-1] if len(ok) else 0 #last = most sensitive threshold 
        out[f"tpr_at_fpr_{t}"] = float(tpr[i]) #attacs we detect
        out[f"actual_fpr_{t}"] = float(fpr[i]) #real fpr
    return out

'''in-domain and cross-domain anomaly detection sweep'''
def main():
    df = load_common()
    rows = []
    n_src = df["source"].nunique()
    prog = Progress(2 * n_src, phase="anomaly_sweep") #2 scenarios per source
    for scenario in ("in_domain", "cross_domain"):
        for target in sorted(df["source"].unique()):
            prog.update(stage=scenario, dataset=target, model="IsolationForest",message=f"{scenario} {target}")
            #in-domain: train and test on benign of target
            if scenario == "in_domain":
                d = df[df["source"] == target]
                tr, te = train_test_split(d, test_size=0.30, random_state=RANDOM_STATE,stratify=d["binary"])
                ben = tr[tr["binary"] == 0] #no leakage only benign train rows
                if len(ben) < 50:
                    continue #too little benign to train
                ben = ben.sample(min(BENIGN_CAP, len(ben)), random_state=RANDOM_STATE)
            #cross-domain: train on benign of other sources, test on all of target
            else:
                ben = df[(df["source"] != target) & (df["binary"] == 0)]
                if len(ben) < 50:
                    continue
                #balance benign training rows from sources
                ben = ben.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), BENIGN_CAP // 3), random_state=RANDOM_STATE))
                te = df[df["source"] == target] 
            s, y, sc, clf = fit_scores(ben, te)
            base = {"scenario": scenario, "dataset": target, "n_benign_train": len(ben)}
            #threshold-independent metrics
            m = metrics_from_scores(s, y)
            rows.append({**base, "contamination": "auto (ROC)", **m})
            log.info("  %-13s %-11s ROC-AUC=%.4f  TPR@1%%FPR=%.3f  TPR@0.1%%FPR=%.3f", scenario, target, m["roc_auc"], m["tpr_at_fpr_0.01"], m["tpr_at_fpr_0.001"])
            #operating-threshold sweep for different contamination levels
            for c in CONTAMINATIONS:
                thr = np.quantile(s, 1 - c)
                pred = s >= thr
                det = float(pred[y == 1].mean()) if (y == 1).any() else None  #attacks detected
                fp = float(pred[y == 0].mean()) if (y == 0).any() else None   #benign misclassified
                #detection_rate is bounded by c fpr_over_c is discriminative measure, <1 = benign pushed to low scores, >1 = detector prefers benign 
                rows.append({**base, "contamination": c, "roc_auc": m["roc_auc"],"detection_rate": det, "fpr_benign": fp,
                             "fpr_over_c": (fp / c) if fp is not None else None, "n_test": m["n_test"], "attack_share": m["attack_share"]})
            del s, y, sc, clf
            
            gc.collect()
            prog.advance()
    #output results to CSV and JSON
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_DIR / "table_anomaly_sweep.csv", index=False)
    (RESULTS_DIR / "anomaly_sweep.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    #summarize ROC-AUC and TPR FPR for each scenario and dataset
    roc = d[d.contamination == "auto (ROC)"]
    log.info("\nROC-AUC independent:\n%s",roc.pivot_table(index="dataset", columns="scenario",values="roc_auc").round(4).to_string())
    log.info("\nPercentage 1%% fake alarms:\n%s",roc.pivot_table(index="dataset", columns="scenario",values="tpr_at_fpr_0.01").round(4).to_string())
    prog.finish(f"anomaly_sweep: {len(rows)} rows")
    log.info("Completed: %d rows", len(rows))


if __name__ == "__main__":
    main()
