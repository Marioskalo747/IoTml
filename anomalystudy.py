#Libraries
import gc
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import COMMON_FEATURES, FIGURES_DIR, RANDOM_STATE, RESULTS_DIR
from progress import Progress


CONTAMINATION = 0.02 #2% of benign is treated as anomalous set FPR
BENIGN_CAP = 60000 #max benign training rows
OUT = RESULTS_DIR / "anomaly_results.json"
#log
log = logging.getLogger("anomaly")
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "anomaly.log", encoding="utf-8")])

'''load combined and cleane features'''
def load_common():
    cache = RESULTS_DIR / "common_cache"
    frames = [pd.read_pickle(p) for p in sorted(cache.glob("*.pkl"))] #deterministic order
    df = pd.concat(frames, ignore_index=True)
    df[COMMON_FEATURES] = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    #the supervised pipeline deduplicates before it splits
    n_raw = len(df)
    df = df[~df.duplicated(subset=COMMON_FEATURES + ["label"])].reset_index(drop=True)
    log.info("dedup: %d -> %d rows (%.1f%% exact duplicates dropped)", n_raw, len(df), 100 * (1 - len(df) / max(n_raw, 1)))
    return df

'''fits scaler amd isolation forest on benign training data'''
def fit_iforest(benign):
    scaler = StandardScaler().fit(benign[COMMON_FEATURES]) #statistics from benign only
    X = scaler.transform(benign[COMMON_FEATURES]) #fixed contamination 
    clf = IsolationForest(n_estimators=200, contamination=CONTAMINATION,random_state=RANDOM_STATE, n_jobs=-1).fit(X)
    return scaler, clf

'''boolean mask (true or -1 for anomalous)'''
def score(scaler, clf, df):
    return clf.predict(scaler.transform(df[COMMON_FEATURES])) == -1


'''anomaly detection study: in-domain and cross-domain (zero-day)'''
def main():
    df = load_common()
    results = []
    sources = sorted(df['source'].unique())
    prog = Progress(max(len(sources) * 2, 1), phase='anomaly_study') #2 scenarios per dataset
    prog.update(stage='loading', message=f'{len(sources)} datasets, two scenarios each')
    i_ds = 0
    # in-domain: train and test on same dataset
    for ds in sorted(df["source"].unique()):
        i_ds += 1   
        prog.update(stage="in_domain", dataset=ds, task="in_domain", model="IsolationForest",substage=f"dataset {i_ds} of {len(sources)}", message=f"in-domain: {ds}")
        d = df[df["source"] == ds]
        tr, te = train_test_split(d, test_size=0.30, random_state=RANDOM_STATE, stratify=d["binary"])
        ben_tr = tr[tr["binary"] == 0] #train on benign only on train
        if len(ben_tr) < 50:
            log.warning(f"Skipping {ds}: not enough benign training samples ({len(ben_tr)})")
            continue #too few benign samples to train
        ben_tr = ben_tr.sample(min(BENIGN_CAP, len(ben_tr)), random_state=RANDOM_STATE)
        scaler, clf = fit_iforest(ben_tr)
        att = te[te["binary"] == 1] #attack in test
        ben = te[te["binary"] == 0]#benign in test
        det = float(score(scaler , clf, att).mean()) if len(att) else None #recall
        fpr = float(score(scaler , clf, ben).mean()) if len(ben) else None #false positive rate
        per_class = {} #per class type
        for lab, g in att.groupby("label"):
            per_class[lab] = float(score(scaler , clf, g).mean())
        results.append({"protocol": "in_domain", "dataset": ds, "n_benign_train": len(ben_tr), "detection_rate": det, "fpr_benign": fpr, "per_class_detection": per_class})
        log.info("[A] %s: detection=%.3f, fpr=%.3f, per_class=%s", ds, det or -1, fpr or -1, per_class)
        prog.advance()  
        gc.collect()
    #cross-domain: train on all other datasets, test on held-out dataset    
    i_ho = 0   
    for held in sorted(df["source"].unique()):
        i_ho += 1  
        prog.update(stage="cross_domain", dataset=held, task="cross_domain", model="IsolationForest",substage=f"holdout {i_ho} of {len(sources)}", message=f"zero-day: {held}")
        ben_src = df[(df["source"] != held) & (df["binary"] == 0)]
        if len(ben_src) < 50:
            continue #balance benign rows across datasets and skip if not enough benign samples
        ben_src = (ben_src.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), BENIGN_CAP //3), random_state=RANDOM_STATE)))
        scaler, clf = fit_iforest(ben_src)
        tgt = df[df["source"] == held] #unseen datasets
        att = tgt[tgt["binary"] == 1]
        ben = tgt[tgt["binary"] == 0]
        det = float(score(scaler , clf, att).mean()) if len(att) else None
        fpr = float(score(scaler , clf, ben).mean()) if len(ben) else None
        per_class = {lab: float(score(scaler , clf, g).mean()) for lab, g in att.groupby("label")}
        results.append({"protocol": "cross_domain", "dataset": held, "n_benign_train": len(ben_src), "detection_rate": det, "fpr_benign": fpr, "per_class_detection": per_class})
        log.info("[B] %s: detection=%.3f, fpr=%.3f, per_class=%s", held, det or -1, fpr or -1, per_class)
        prog.advance() 
        gc.collect()
        
    #persist results and plot    
    OUT.write_text(json.dumps(results, indent=1, default=str))
    rows = []
    for r in results:
        rows.append({"dataset": r["dataset"], "protocol": "Same dataset (in-domain)" if r["protocol"] == "in_domain" else "Unseen dataset (cross-domain)", "detection_rate": r["detection_rate"], "fpr_benign": r["fpr_benign"]})
    d = pd.DataFrame(rows).melt(id_vars=["dataset", "protocol"], var_name="metric", value_name="rate")
    g = sns.catplot(data=d, x="dataset", y="rate", hue="metric", col="protocol", kind="bar", height=4, aspect=1.15) #one panel per protocol 
    g.set_axis_labels("", "Rate")
    g.fig.suptitle("Only benign train (Isolation Forest)", y=1.04)
    g.fig.savefig(FIGURES_DIR / "anomaly_detection.png", bbox_inches="tight", dpi=300)
    log.info("anomaly detection study complete: %d records", len(results))
    prog.finish(f"anomaly study complete: {len(results)} records")   
        
if __name__ == "__main__":
    main()
    
    
            
            
    
    