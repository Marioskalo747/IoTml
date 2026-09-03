#Libraries
import gc
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMON_FEATURES, RANDOM_STATE, RESULTS_DIR, TEST_SIZE
from evaluation import evaluate_model
from loaders import available_datasets
from progress import Progress

LODO_TRAIN_CAP = 40000 #max rows per class for LODO       
LODO_TREES = 150 #fewer trees for LODO to speed up training             
log = logging.getLogger("domain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler(RESULTS_DIR / "domain_shift_study.log", encoding="utf-8")])

'''load combined and clean features'''
def load_common():
    cache = RESULTS_DIR / "common_cache"
    frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
    df = pd.concat(frames, ignore_index=True)
    df[COMMON_FEATURES] = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    return df

'''Z-scores features per source (dataset)(remove scale differences)'''
def normalise_per_source(df, cols=COMMON_FEATURES):
    out = df.copy() #copy original df
    #each source (dataset) is normalised independently to remove scale differences
    for s, idx in df.groupby("source").groups.items():
        out.loc[idx, cols] = StandardScaler().fit_transform(df.loc[idx, cols])
    return out

'''measure source identifiability'''
def source_identifiability(df):
    n_raw = len(df)
    df = df[~df.duplicated(subset=COMMON_FEATURES + ["label"])]
    log.info("source identifiability dedup: %d -> %d rows (%.1f%% exact duplicates dropped)", n_raw, len(df), 100 * (1 - len(df) / max(n_raw, 1)))
    X, src = df[COMMON_FEATURES], df["source"] #remove suspicious features that may leak dataset identity
    subsets = {"all_11": COMMON_FEATURES,"no_ports_9": [c for c in COMMON_FEATURES if "port" not in c],"no_ports_no_time_6": [c for c in COMMON_FEATURES
        if "port" not in c and c not in ("duration", "pkt_rate", "byte_rate")],"sizes_only_3": ["total_pkts", "total_bytes", "avg_pkt_size"]} #share of the largest dataset
    out = {"chance_level": float(src.value_counts(normalize=True).max())}
    for tag, cols in subsets.items():
        #target is dataset label
        a, b, ya, yb = train_test_split(X[cols], src, test_size=0.3, random_state=RANDOM_STATE, stratify=src)
        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE).fit(a, ya)
        acc = float((m.predict(b) == yb).mean())
        #top 3 features that betray the dataset of origin
        out[tag] = {"accuracy": acc, "n_features": len(cols),"top_features": pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False).head(3).round(3).to_dict()}
        log.info("source identifiability  %-20s accuracy=%.4f", tag, acc)
        del m
        gc.collect()
    return out

'''Leave-One-Dataset-Out (LODO) cross-domain evaluation (zero-day)'''
def lodo(df, tag):
    rows = []
    for held in sorted(df["source"].unique()):
        train = df[df["source"] != held]
        test = df[df["source"] == held]
        shared = set(train["label"]) & set(test["label"]) #shared classes only
        train = train[train["label"].isin(shared)]
        test = test[test["label"].isin(shared)]
        if test.empty or train.empty or len(shared) < 2:
            continue #on less than 2 shared classes cannot evaluate
        tr = train.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), LODO_TRAIN_CAP), random_state=RANDOM_STATE))
        clf = RandomForestClassifier(n_estimators=LODO_TREES, random_state=RANDOM_STATE,n_jobs=-1).fit(tr[COMMON_FEATURES], tr["label"])
        labels = sorted(shared)
        r = evaluate_model(clf, test[COMMON_FEATURES], test["label"], labels) 
        #tag separates raw vs per source normalised features
        rows.append({"experiment": f"lodo_{tag}", "holdout": held,"n_train": len(tr), "n_test": len(test), "n_shared_classes": len(shared), **{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")}})
        log.info("  LODO %-12s holdout=%-11s f1_macro=%.4f", tag, held, r["f1_macro"])
        del clf
        gc.collect()
    return rows

'''all datasets pooled with random split'''
def combined(df, tag):
    #drop duplicate rows before splitting to avoid leakage (same flow in train and test)
    n_raw = len(df)
    df = df[~df.duplicated(subset=COMMON_FEATURES + ["label"])]
    log.info("  combined %-12s dedup: %d -> %d rows (%.1f%% duplicates dropped)", tag, n_raw, len(df), 100 * (1 - len(df) / max(n_raw, 1)))
    X = df[COMMON_FEATURES]
    y = df["label"].astype(str)
    #random split (train test contain trafic from same networks)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,random_state=RANDOM_STATE).fit(Xtr, ytr)
    labels = sorted(pd.unique(y))
    r = evaluate_model(clf, Xte, yte, labels)
    log.info("  combined %-12s f1_macro=%.4f", tag, r["f1_macro"])
    del clf
    gc.collect()
    #same record structure as LODO for easier comparison
    return {"experiment": f"combined_{tag}", "holdout": "-", "n_train": len(Xtr),"n_test": len(Xte), "n_shared_classes": len(labels),**{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")}}

'''Full domain shift study: source identifiability, LODO (raw and normalised), combined (raw and normalised)'''
def main():
    df = load_common()
    prog = Progress(5, phase="domain_shift_study")
    prog.update(stage="loading", message=f"{len(df):,} rows, {df['source'].nunique()} datasets")
    log.info("loaded %d rows from %d datasets", len(df), df["source"].nunique())
    #how much dataset identity can be inferred from the features (dataset leakage)
    prog.update(stage="source_id", message="source identifiability")
    log.info("1.Source identifiability")
    ident = source_identifiability(df)
    prog.advance()
    #LODO on raw features
    log.info("2. LODO (Leave-One-Dataset-Out)")
    prog.update(stage="lodo_raw", message="LODO with raw features")
    rows = lodo(df, "raw")
    prog.advance()
    #LODO after normalising features per source (dataset) to remove scale differences
    log.info("3.LODO normalised features")
    prog.update(stage="lodo_norm", message="LODO with normalised features")
    dfn = normalise_per_source(df)
    #NOT a zero-day protocol
    rows += lodo(dfn, "per_source_normalised_transductive")
    prog.advance()
    #combined (raw and normalised)
    log.info("4. combined (raw and normalised)")
    rows.append(combined(df, "raw"))
    rows.append(combined(dfn, "per_source_normalised_transductive"))
    #json and csv output
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_DIR / "table_domain_shift.csv", index=False)
    #source identifiability table
    ident_rows = []
    chance = ident.get("chance_level")
    for tag, v in ident.items():
        if tag == "chance_level":
            continue
        ident_rows.append({"feature_subset": tag, "n_features": v["n_features"], "accuracy": v["accuracy"], "chance_level": chance,
                           "lift_over_chance": v["accuracy"] - chance, "top_features": "; ".join(f"{k}={val}" for k, val in v["top_features"].items())})
    if ident_rows:
        pd.DataFrame(ident_rows).to_csv(RESULTS_DIR / "table_source_identifiability.csv", index=False)
        log.info("\nsource identifiability (chance=%.4f):\n%s", chance,
                 pd.DataFrame(ident_rows)[["feature_subset", "n_features", "accuracy", "lift_over_chance"]].round(4).to_string(index=False))
    (RESULTS_DIR / "domain_shift_study.json").write_text(
        json.dumps({"source_identifiability": ident, "experiments": rows},indent=1, default=str), encoding="utf-8")
    #F1 gained by normalising features per dataset
    piv = d[d.experiment.str.startswith("lodo")].pivot_table(index="holdout", columns="experiment", values="f1_macro")
    if not piv.empty:
        piv["recovery"] = piv.get("lodo_per_source_normalised", 0) - piv.get("lodo_raw", 0)
        log.info("\nLODO f1_macro per held-out dataset:\n%s", piv.round(4).to_string())
    log.info("\ncombined:\n%s", d[d.experiment.str.startswith("combined")][["experiment", "f1_macro", "accuracy"]].round(4).to_string(index=False))
    prog.finish(f"domain_shift: {len(rows)} experiments")
    log.info("Completed: %d experiments", len(rows))


if __name__ == "__main__":
    main()
