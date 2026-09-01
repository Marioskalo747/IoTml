#Libraries
import gc
import json
import sys
import time
import traceback
import os
from pathlib import Path    
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from smoke_data import smoke_datasets, SMOKE_DATA_DIR
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

from config import (COMMON_FEATURES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE,
                    SLOW_MODEL_TRAIN_CAP, MIN_ROWS_PER_CLASS_SLOW)
from evaluation import (evaluate_model, class_distribution, plot_confusion_matrix, feature_importance, model_comparison)
from loaders import available_datasets, DATASETS
from models_zoo import model_zoo
from preprocessing import build_xy, split, make_preprocessor, split_meta
from progress import Progress



PROGRESS : Progress | None = None
RESULTS = [] #Experiment results of run to all_results.json
EXISTING = {} #previous results loaded

'''Identifier for an experiment record'''
def tag(dataset, task, model):
    return f"{dataset}__{task}__{model}"

'''Atomic record for an experiment result'''
def checkpoint():
    tmp = RESULTS_DIR / "all_results.tmp"
    with open(tmp, "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    for _try in range(5):   
        try:                
            tmp.replace(RESULTS_DIR / "all_results.json")
            break          
        except PermissionError:
            time.sleep(0.3)

'''Stratified subsample of a training set for slow models'''   
def subsample(X_train, y_train, cap):
    if len(X_train) <= cap:
        return X_train, y_train
    #the floor is the point
    idx = (pd.Series(range(len(X_train)), index=X_train.index).groupby(y_train, group_keys=False)
           .apply(lambda x: x.sample(n=min(len(x), max(round(len(x)*cap/len(X_train)), MIN_ROWS_PER_CLASS_SLOW)),
                                     random_state=RANDOM_STATE)))
    return X_train.loc[idx.index], y_train.loc[idx.index]

'''Fingerprint of a training set for caching and reuse of models'''
def data_fingerprint(X_train, y_train):
    #slow_model_train_cap belongs in the fingerprint
    parts = [str(len(X_train)), ",".join(map(str, sorted(X_train.columns))), ",".join(f"{k}:{v}" for k, v in sorted(pd.Series(y_train).value_counts().items())), f"slowcap={SLOW_MODEL_TRAIN_CAP}"]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16]

'''Train all models in the zoo for a dataset (produce metrics, models, figures)'''
def train_all_models(X_train, X_test, y_train, y_test, dataset, task):
    labels = sorted(pd.unique(pd.concat([y_train, y_test]).astype(str)))
    fingerprint = data_fingerprint(X_train, y_train)
    pre = make_preprocessor(X_train) #preprocessor inside the pipeline to avoid data leakage
    out = []
    n_models_here = len(model_zoo(len(labels)))   
    i_model = 0                                  
    for spec in model_zoo(len(labels)):
        i_model += 1                              
        ta = tag(dataset, task, spec.name)
        prev = EXISTING.get(ta)
        #resume under conditions: previous results exist, same parameters, same training data fingerprint, not combined dataset
        if (prev is not None and dataset != "combined" and (MODELS_DIR / f"{ta}.joblib").exists() and prev.get("params") == spec.params and prev.get("data_fingerprint") == fingerprint):     
            print(f"resume {ta} previous results kept")
            RESULTS.append(prev)
            out.append(prev)
            if PROGRESS:
                PROGRESS.update(model=spec.name, message=f" {ta} reuse")
                PROGRESS.advance()
            continue
        
        if PROGRESS:
            PROGRESS.update(stage="training", dataset=dataset, task=task, model=spec.name, substage=f"model {i_model} of {n_models_here} - {len(labels)} classes", message=f"training {spec.name} on {dataset} {task}")
        try:
            Xt, yt = (X_train, y_train) #slow model subsample
            if spec.slow:
                Xt, yt = subsample(X_train, y_train, SLOW_MODEL_TRAIN_CAP)
            pipeline = Pipeline([("pre", pre), ("clf", spec.build())])
            t0 = time.perf_counter()
            pipeline.fit(Xt, yt.astype(str))
            train_time = time.perf_counter() - t0
            proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    proba = pipeline.predict_proba(X_test) #needed for ROC-AUC and lod-loss
                except Exception:
                    proba = None #expose no probabilities
            res = evaluate_model(pipeline, X_test, y_test.astype(str), labels, proba=proba)
            #metadata as identity, cost, sizes and fingerprint for the metrics
            res.update({"dataset": dataset, "task": task, "model": spec.name, "family": spec.family, "params": spec.params, "train_time_s": train_time, "train_rows": len(Xt), "test_rows": len(X_test), "n_features": pipeline.named_steps["pre"].transform(X_test.head(1)).shape[1], "data_fingerprint": fingerprint,
                        "train_rows_available": len(X_train), "capped": bool(len(Xt) < len(X_train)), "train_cap": SLOW_MODEL_TRAIN_CAP if spec.slow else None})
            #which split protocol this dataset actually got, and whether the test set can carry a macro average at all
            res.update(split_meta(dataset, y_train, y_test))
            #what this model actually saw of its rarest class stays available
            res["train_min_class_rows"] = int(pd.Series(yt).value_counts().min())
            #wether iterative learners converge or they stop because they ran out of budget
            _clf = pipeline.named_steps["clf"]
            _inner = getattr(_clf, "base", _clf)
            _ni = getattr(_inner, "n_iter_", None)
            _ni = int(np.max(_ni)) if _ni is not None else None
            _mx = getattr(_inner, "max_iter", None)
            res["n_iter"] = _ni
            res["converged"] = None if _ni is None or _mx is None else bool(_ni < _mx)
            RESULTS.append(res)
            out.append(res)
            joblib.dump(pipeline, MODELS_DIR / f"{ta}.joblib", compress=3) #whole pipeline saved
            
            plot_confusion_matrix(res, f"{spec.name} on {dataset} ({task})", FIGURES_DIR / f"confusion_{ta}.png")
            clf = pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"): #only tree models have importances
                try:
                    names = pipeline.named_steps["pre"].get_feature_names_out()
                    feature_importance(names, clf.feature_importances_,dataset, spec.name, FIGURES_DIR / f"feature_importance_{ta}.png")
                except Exception:
                    pass
            msg = (f"{ta} accuracy={res['accuracy']:.4f}, f1_macro={res['f1_macro']:.4f}, train_time_s={train_time:.1f}s")
            print(msg)
            if PROGRESS:
                PROGRESS.update(message=f"OK {msg}")
        except MemoryError:
            #run out of memory, skip this model
            print(f"MemoryError: skipping {ta}")
            gc.collect()
        except Exception as e:
            print(f"Exception: skipping {ta}: {e}")
            traceback.print_exc(limit = 1) #one line of traceback to avoid flooding the console
            if PROGRESS:
                PROGRESS.update(message=f"FAIL {ta}: {e}")
        finally:
            #guaranteed progress and checkpoint even after fail
            if PROGRESS:
                PROGRESS.advance()
            checkpoint()
            gc.collect()
    return out

'''Full processing of a dataset: load, preprocess, train all models'''
def run_dataset(name, loader, to_common, collect_common: list):
    print(f"running {name}")
    if PROGRESS:
        PROGRESS.update(stage="loading", dataset=name, task="", model="", message=f"loading {name}")
    t0 = time.perf_counter()
    df = loader()
    print(f"loaded {len(df):,} rows in {time.perf_counter() - t0:.0f}s, classes={dict(df['label'].value_counts())}")    
    if PROGRESS:
        PROGRESS.update(message=f"preprocessing {name}: {len(df):,} rows in {time.perf_counter() - t0:.0f}s", details={"preprocessing": {
            "dataset": name, "rows": len(df), "load_time": round(time.perf_counter() - t0, 1), "classes": {str(k): int(v) for k, v in df['label'].value_counts().items()}}})
    #conver to common schema (11 features)
    #each later study reads data from here (adaption, anomaly, domain shift)
    common = to_common(df).assign(source=name)
    cache = RESULTS_DIR / "common_cache"
    cache.mkdir(exist_ok=True, parents=True)
    common.to_pickle(cache / f"{name}.pkl")
    collect_common.append(common)
    X, y, y_bin = build_xy(df)
    del df
    gc.collect()
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = split(X, y, y_bin, dataset=name)
    #plotted AFTER the split so the figure shows the distribution the scores actually refer to
    class_distribution(y_train, name, FIGURES_DIR / f"class_distribution_{name}.png", y_test=y_test)
    if PROGRESS:
        PROGRESS.update(details={"preprocessing": {"dataset": name, "n_features": X.shape[1], "train_rows": len(X_train), "test_rows": len(X_test), "n_classes": int(y.nunique())}})
    results = []
    results += train_all_models(X_train, X_test, y_train,  y_test, name, "multiclass") 
    #same split, different target for binary classification (Benign vs Attack)
    yb_train_s = y_bin_train.map({0: "Benign", 1: "Attack"})
    yb_test_s = y_bin_test.map({0: "Benign", 1: "Attack"})
    results += train_all_models(X_train, X_test, yb_train_s, yb_test_s, name, "binary")
    cmp_df = pd.DataFrame([{k: r[k] for k in ("model", "accuracy", "f1_macro", "f1_weighted", "balanced_accuracy")} for r in results if r["task"] == "multiclass"])
    
    if not cmp_df.empty:
        model_comparison(cmp_df, name, FIGURES_DIR / f"model_comparison_{name}.png")
    
    gc.collect()
    return results

'''LODO (Leave-One-Dataset-Out) cross-dataset evaluation'''
def run_cross_dataset(df):
    out = []
    df = df.copy()
    df[COMMON_FEATURES] = (df[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    holdouts = sorted(df["source"].unique())   
    i_hold = 0                                 
    for held in sorted(df["source"].unique()):
        i_hold += 1                            
        if PROGRESS: 
            PROGRESS.update(stage="LODO", dataset=held, task="cross_dataset", model="RandomForest",substage=f"holdout {i_hold} of {len(holdouts)}",message=f"LODO without {held}")
        train = df[df["source"] != held]
        test = df[df["source"] == held]
        shared = set(train["label"]) & set(test["label"]) #shared classes only
        train = train[train["label"].isin(shared)]
        test = test[test["label"].isin(shared)]
        if test.empty or train.empty or len(shared) < 2:
            continue #fewer than 2 classes
        #40k cap per class
        tr = train.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), 40000), random_state=RANDOM_STATE))
        clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1).fit(tr[COMMON_FEATURES], tr["label"])
        labels = sorted(shared)
        res = evaluate_model(clf, test[COMMON_FEATURES], test["label"], labels=labels)
        #name convention (LODO_holdout_X) for later scripts
        res.update({"dataset": f"LODO_holdout_{held}", "task": "cross_dataset", "model": "RandomForest", "family": "ensemble", "params": {"n_estimators": 150}, "train_time_s": None, "train_rows": len(tr), "test_rows": len(test), "n_features": len(COMMON_FEATURES)})
        RESULTS.append(res)
        out.append(res)
        checkpoint()
        print(f"LODO {held:12s}: accuracy={res['accuracy']:.4f}, f1_macro={res['f1_macro']:.4f}")
    return out

'''Combined model on all datasets (multiclass and binary)'''
def run_combined(common_frames):
    if len(common_frames) < 2:
        print("not enough datasets for combined model")
        return [] #one dataset
    print(f"running combined model ")
    df = pd.concat(common_frames, ignore_index=True)
    print(f"combined dataset has {len(df):,} rows from {df['source'].nunique()} datasets")
    if PROGRESS:
        PROGRESS.update(stage="loading", dataset="combined", task="", message=f"preprocessing combined {len(df):,} rows from {df['source'].nunique()} datasets")
    #only 11 common features
    X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    y = df["label"].astype(str)
    y_bin = df["binary"].astype(int)
    X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = split(X, y, y_bin, dataset="combined")
    class_distribution(y_train, "combined", FIGURES_DIR / "class_distribution_combined.png", y_test=y_test)
    results = []
    results += train_all_models(X_train, X_test, y_train,  y_test, "combined", "multiclass")
    yb_train_s = y_bin_train.map({0: "Benign", 1: "Attack"})
    yb_test_s = y_bin_test.map({0: "Benign", 1: "Attack"})
    results += train_all_models(X_train, X_test, yb_train_s, yb_test_s, "combined", "binary")
    cmp_df = pd.DataFrame([{k: r[k] for k in ("model", "accuracy", "f1_macro", "f1_weighted", "balanced_accuracy")} for r in results if r["task"] == "multiclass"])
    if not cmp_df.empty:
        model_comparison(cmp_df, "combined", FIGURES_DIR / "model_comparison_combined.png")
    results += run_cross_dataset(df) #LODO runs on same pooled data
    return results

'''Pick the best model'''
def pick_best(results):
    best = {}
    for task in ("multiclass", "binary"):
        cands = [r for r in results if r["dataset"] == "combined" and r["task"] == task]
        if cands:
            top = max(cands, key=lambda r: r["f1_macro"]) #criterion macro_F1
            #path is the filename (easier deployment loading)
            best[task] = {"model": top["model"], "path": f"combined__{task}__{top['model']}.joblib", "f1_macro": top["f1_macro"], "accuracy": top["accuracy"], "labels": top["labels"]}
    per_ds = {}
    for r in results:
        if r["task"] != "multiclass" or str(r["dataset"]).startswith("LODO"):
            continue
        cur = per_ds.get(r["dataset"])
        if cur is None or r["f1_macro"] > cur["f1_macro"]: #single champion
            per_ds[r["dataset"]] = {"model": r["model"], "f1_macro": r["f1_macro"], "accuracy": r["accuracy"], "path": f"{r['dataset']}__{r['task']}__{r['model']}.joblib"}
    best["per_dataset"] = per_ds
    return best

'''Central training pipeline combined+LODO with resume and checkpointing'''
def main():
    global PROGRESS
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    
    args = [a for a in sys.argv[1:] if not a.startswith("-")] #non=dash arguments
    smoke = "--smoke" in sys.argv
    #The pooled experiments (combined + LODO) are built from common_cache, which carries the eleven features and no TS_COL
    no_pooled = "--no-pooled" in sys.argv
    if smoke:
        smoke_datasets() #tiny synthetic datasets
        os.environ["IOT_IDS_DATA_DIR"] = str(SMOKE_DATA_DIR) #config reads from there (no real data)
    
    available = available_datasets()
    wanted = [a for a in (args or available) if a in DATASETS]
    if not wanted:
        print("No datasets found. Place them under the dataset dir (see config.py)")
        sys.exit(1)
    print(f"running datasets: {wanted}")        
    #resume previous results if any
    existing = []
    res_path = RESULTS_DIR / "all_results.json"
    if res_path.exists():
        try:
            existing = json.loads(res_path.read_text())
        except Exception:
            existing = [] #existing
    for r in existing:
        if (not str(r["dataset"]).startswith("LODO") and r["dataset"] != "combined" and r["task"] in ("multiclass", "binary")):
            EXISTING[tag(r["dataset"], r["task"], r["model"])] = r
    #kept previous results
    kept = [r for r in existing if r["dataset"] not in wanted and r["dataset"] != "combined" and not str(r["dataset"]).startswith("LODO")]
    kept_names = sorted({r["dataset"] for r in kept})
    if kept_names:
        print(f"keeping previous results for datasets: {kept_names}")
    RESULTS.extend(kept)
    
    n_models = len(model_zoo(3)) #model count does not depend on class count
    total_steps = (len(wanted)+1) * n_models * 2 #+1 for combined, x2 for multiclass+binary
    PROGRESS = Progress(total_steps, phase="run_all")
    PROGRESS.update(stage="starting", message=f"running {wanted}, {n_models} models x2 each")
    
    common_frames = []
    for name in wanted:
        loader, to_common = DATASETS[name]
        try:
            run_dataset(name, loader, to_common, common_frames)
        except FileNotFoundError as e:
            print(f"skipping {name}: {e}")
            PROGRESS.update(message=f"skipping {name}: {e}") #dataset file is missing
        except Exception as e:
            print(f"skipping {name}: {e}")
            traceback.print_exc(limit = 3)
            PROGRESS.update(message=f"skipping {name}: {e}")
        checkpoint()
    #combined model needs all datasets
    cache_dir = RESULTS_DIR / "common_cache"
    have = {f["source"].iloc[0] for f in common_frames if len(f)} #datasets already loaded
    for name in kept_names:
        if name in have or name not in DATASETS:
            continue
        pkl = cache_dir / f"{name}.pkl"
        try:
            if pkl.exists():
                common_frames.append(pd.read_pickle(pkl)) #cheap reload from cache
                print(f"reloaded {name} from cache")
            else:
                #without cache (reload dataset)
                print(f"skipping {name}: no cache")
                PROGRESS.update(stage="loading", dataset=name, message=f"loading {name} for combined")
                loader, to_common = DATASETS[name]
                df = loader()
                common = to_common(df).assign(source=name)
                common.to_pickle(pkl)
                common_frames.append(common)
                del df
                gc.collect()
        except Exception as e:
            print(f" skipping {name}: {e}")
    if no_pooled:
        print("skipping combined + LODO (--no-pooled): they cannot be run under a non-random "
              "protocol and would be built from an incomplete pool")
    else:
        try:
            run_combined(common_frames)
        except Exception as e:
            print(f"combined model failed: {e}")
            traceback.print_exc(limit = 3)
    checkpoint()
    #tuning.py appends 32 tuned records to all_results.json 
    best = pick_best(RESULTS)
    with open(MODELS_DIR / "best_models.json", "w") as f:
        json.dump(best, f, indent=1) #pointer for best models
    PROGRESS.finish(f"completed {len(RESULTS)} results, best models ")
    print(f"\nDone. {len(RESULTS)} experiments. Results in {RESULTS_DIR}")
    
if __name__ == "__main__":
    main()
    
    
