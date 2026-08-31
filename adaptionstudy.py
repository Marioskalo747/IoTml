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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #render charts to files
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import COMMON_FEATURES, FIGURES_DIR, RANDOM_STATE, RESULTS_DIR 
from progress import Progress
from evaluation import evaluate_model

#Fractions of local data added to training 
FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
SOURCE_CAP = 40000 #rows per class from datasets

OUT = RESULTS_DIR / "adaptation_results.json"

#log output to both console and file
log = logging.getLogger("adapt")
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(RESULTS_DIR / "adaptation.log",encoding="utf-8")])


'''loads combinred dataset from cache and cleans features'''
def load_common():
    cache = RESULTS_DIR / "common_cache"
    frames = [pd.read_pickle(p) for p in sorted(cache.glob("*.pkl"))]
    df = pd.concat(frames, ignore_index=True)
    df[COMMON_FEATURES] = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    return df

'''Adaptation study LODO: trafic is needed to make model adapt to new network'''
def main():
    df = load_common()
    results = []
    #resume from previous run, but only records produced with the deduplicated protocol:
    #older records came from a split that allowed duplicate flows in train and test
    if OUT.exists():
        try:
            results = [r for r in json.loads(OUT.read_text()) if r.get("dedup")]
        except Exception:
            results = []
    done = {(r["holdout"], r["fraction"]) for r in results}
    holdouts = sorted(df["source"].unique())
    #progress
    prog = Progress(max(len(holdouts) * len(FRACTIONS), 1), phase="adaptation_study")
    prog.update(stage="loading", message=f"{len(holdouts)} networks x {len(FRACTIONS)} fractions")
    i_ho = 0
    #one target network at a time (train on the rest and adapt)
    for held in sorted(df["source"].unique()):
        i_ho += 1
        prog.update(stage="adaptation", dataset=held, task="cross_dataset", model="RandomForest",substage=f"target network {i_ho} of {len(holdouts)}", message=f"adapting to {held}")
        src = df[df["source"] != held] #Other networks as source
        tgt = df[df["source"] == held] #Target network as target
        #duplicate flows are dropped BEFORE the target split: without this the small "local"
        #sample contains exact copies of test rows and the adaptation gain is memorisation
        n_raw = len(tgt)
        tgt = tgt[~tgt.duplicated(subset=COMMON_FEATURES + ["label"])]
        log.info("target %s: %d -> %d rows after dedup (%.1f%% duplicates dropped)", held, n_raw, len(tgt), 100 * (1 - len(tgt) / max(n_raw, 1)))
        #target split and downsampling to shared classes
        tgt_pool, tgt_test = train_test_split(tgt, test_size=0.30, random_state=RANDOM_STATE, stratify=tgt["label"])
        shared = set(src["label"]) & set(tgt_test["label"]) 
        src = src[src["label"].isin(shared)]
        src = src.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), SOURCE_CAP), random_state=RANDOM_STATE))
        test = tgt_test[tgt_test["label"].isin(shared)]
        labels = sorted(shared)
        
        #one adaptation fraction at a time
        for frac in FRACTIONS:
            prog.update(substage=f"{frac * 100:g}% local traffic")
            if(held, frac) in done:
                continue
            #per class sampling
            local = (tgt_pool.groupby("label", group_keys=False).apply(lambda x: x.sample(max(1, int(len(x) * frac)), random_state=RANDOM_STATE)) if frac>0 else tgt_pool.iloc[:0])
            local = local[local["label"].isin(shared)]
            train = pd.concat([src, local], ignore_index=True)
            clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1).fit(train[COMMON_FEATURES], train["label"]) #evaluate model on the target network
            res = evaluate_model(clf, test[COMMON_FEATURES], test["label"], labels)#evaluate model on the target network
            rec = {"holdout": held, "fraction": frac, "n_local": len(local), "n_source": len(src), "n_test": len(test), "dedup": True, **{k: res[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "cohen_kappa", "g_mean")}} #experiment identifier and results
            #checkpoint, info, progress
            results.append(rec)
            OUT.write_text(json.dumps(results, indent=1, default=str))
            log.info(f"Holdout={held}, fraction={frac:.3f}, n_local={len(local)}, n_source={len(src)}, n_test={len(test)}: {res}") 
            prog.advance()
            #free forest's memory
            del clf
            gc.collect()

    #plot curve per holdout (percentage of local data vs F1 macro)
    d=pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    for held, g in d.groupby("holdout"):
        g = g.sort_values("fraction")
        ax.plot(g["fraction"]*100, g["f1_macro"], marker="o", lw=2, label=held)
    ax.set_xlabel("Fraction of target pool used for adaptation (%)")
    ax.set_ylabel("F1 macro score")
    ax.set_title("Adaptation study LODO")
    ax.grid(alpha=0.3)
    ax.legend(title="Holdout")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "adaptation_study.png", dpi=300)
    log.info("Adaptation study completed. Results saved to %s", len(results))
    prog.finish(f"adaptation study complete: {len(results)} records")


if __name__ == "__main__":
    main()
