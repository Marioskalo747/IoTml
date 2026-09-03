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
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


from config import FIGURES_DIR, RANDOM_STATE, RESULTS_DIR
from evaluation import (evaluate_model, cv_boxes, learning_curves, strategy_comparison)
from loaders import DATASETS, available_datasets
from models_zoo import build_with_class_weight, model_zoo, CLASS_WEIGHT_CAPABLE
from preprocessing import (apply_imbalance, build_xy, split, select_features, make_preprocessor, split_meta)
from progress import Progress

PROGRESS = None #record of progress for the study

CAP = 60000 #row cap for cv part
ABLATION_ROWS = 60000 #training cap for ALL parts
RESAMPLE_MAX = 300000 #ceiling after oversampling/smote
IMBALANCE = ["none", "class_weight", "undersample", "oversample", "smote"]
FS_METHODS = ["none", "variance", "corr95", "kbest_mi"]
SAMPLE_SIZES = [2000, 5000, 10000, 25000, 50000, 100000] #log-spaced sample sizes for learning curve
ABLATION_MODELS = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "TorchMLP"]
LC_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "TorchMLP"] #4 models from different families for learning curve analysis
CV_MODELS = ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"] #strong models only for cross-validation analysis
OUT_PATH = RESULTS_DIR / "extended_results.json"

log = logging.getLogger("extended")
logging.basicConfig( level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(RESULTS_DIR / "extended.log",encoding="utf-8")])

SINK: list[dict] = [] #every record of the study
DONE: set[str] = set() #keys of already completed records
META: dict[tuple, dict] = {}

'''Builds a unique key for each experiment'''
def key(**kwargs):
    return "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))

'''Saves the SINK to disk in a safe manner'''
def save():
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(SINK, indent=1, default=str))
    for _try in range(5):  
        try:               
            tmp.replace(OUT_PATH)
            break           
        except PermissionError:
            time.sleep(0.3)
            
'''Records a result in the SINK and marks it as DONE'''   
def record(kind, extra_metrics, **kwargs):
    rec = {"kind": kind, **kwargs, **META.get((kwargs.get("dataset"), kwargs.get("task")), {}), **extra_metrics}
    SINK.append(rec)
    DONE.add(key(kind=kind, **{k: kwargs[k] for k in ("dataset", "task", "model", "variant")}))
    save() #checkpoint
    if PROGRESS: 
        PROGRESS.advance(model=kwargs.get("model"), substage=f"{kwargs.get('model')} - {kwargs.get('variant')}")
    
'''Stratified sampling with 5-row minimum per class''' 
def subsample(X, y, cap, seed=RANDOM_STATE):
    if len(X) <= cap:
        return X, y
    np.random.seed(seed)
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False).apply(lambda x: x.sample(n=max(5, round(len(x)*cap/len(X))), random_state=seed)))
    return X.loc[idx.index], y.loc[idx.index]

'''Model specification lookup by name and number of classes'''
def spec(name, n_classes):
    for s in model_zoo(n_classes):
        if s.name == name:
            return s
    return None

'''Extracts relevant metrics of interest'''
def metrics(res):
    keys = ("accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc", "cohen_kappa", "g_mean", "predict_us_per_sample") 
    out = {k: res[k] for k in keys if k in res}
    if res.get("roc_auc_ovr") is not None:
        out["roc_auc_ovr"] = res["roc_auc_ovr"]
    return out

'''The main fit-evaluate function (preprocessing + model in a pipeline)'''
def fit_eval(model, X_train, y_train, X_test, y_test, labels):
    pipe = Pipeline([("preprocessor", make_preprocessor(X_train)), ("clf", model)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train.astype(str))
    train_time = time.perf_counter() - t0
    proba = None
    try:
        proba = pipe.predict_proba(X_test)
    except Exception:
        pass #models that don't support predict_proba
    res = evaluate_model(pipe, X_test, y_test.astype(str), labels, proba=proba)
    return res, train_time, pipe

'''Part a 5 imbalance analysis'''
def part_a_imbalance(X_train, Xtest, y_train, y_test, dataset, task):
    labels = sorted(pd.unique(pd.concat([y_train, y_test]).astype(str)))
    n_classes = len(labels)
    if PROGRESS:   
        PROGRESS.update(stage="imbalance", dataset=dataset, task=task, substage="class-imbalance strategies", message=f"[imbalance] {dataset}/{task}")
    rows_for_plot = []
    for strategy in IMBALANCE:
        for model_name in ABLATION_MODELS:
            specif = spec(model_name, n_classes)
            if specif is None:
                continue #not installed library
            k = key(kind="imbalance", dataset=dataset, task=task, model=model_name, variant=strategy)
            if k in DONE:
                continue #already done
            if strategy == "class_weight":
                if model_name not in CLASS_WEIGHT_CAPABLE:
                    continue  #no class_weight support
            try:
                Xa, ya = X_train, y_train
                if strategy in ("undersample", "oversample", "smote"):
                    #imblearn samplers require numeric matrix (one hot encoding)
                    Xa = pd.get_dummies(X_train, dummy_na=False).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    Xa, ya = apply_imbalance(Xa, y_train, strategy=strategy)
                    if len(Xa) > RESAMPLE_MAX:
                        #reset index before sumpling again
                        ya = pd.Series(ya).reset_index(drop=True)
                        Xa = pd.DataFrame(Xa).reset_index(drop=True)
                        Xa, ya = subsample(Xa, ya, RESAMPLE_MAX)
                    #one-hot encoded and religned to train test
                    Xte_use = pd.get_dummies(Xtest, dummy_na=False).reindex(columns=Xa.columns, fill_value=0)
                else:
                    Xte_use = Xtest
                model = (build_with_class_weight(specif) if strategy == "class_weight" else specif.build()) #class_weight on the model
                res, t, _ = fit_eval(model, Xa, ya, Xte_use, y_test, labels)
                m = metrics(res)
                record("imbalance", {**m, "train_time_s": t, "train_rows": len(Xa)}, dataset=dataset, task=task, model=model_name, variant=strategy)
                rows_for_plot.append({"model": model_name, "imbalance": strategy, "f1_macro": m["f1_macro"]})
                log.info("[A] %s/%s %s × %s: f1m=%.4f (%.0fs)", dataset, task, model_name, strategy, m["f1_macro"], t)
            except Exception as e:
                log.exception("[A] FAIL %s/%s %s × %s: %s", dataset, task, model_name, strategy, e)
                
            gc.collect()
    #SINK built figure to cover resume restored results 
    prev = [r for r in SINK if r["kind"] == "imbalance" and r["dataset"] == dataset and r["task"] == task]
    
    if prev:
        strategy_comparison([{"model": r["model"], "imbalance": r["variant"], "f1_macro": r["f1_macro"]} for r in prev], dataset, task, FIGURES_DIR / f"ablation_imbalance_{dataset}_{task}.png")

'''Part b feature selection analysis'''        
def part_b_fs(X_train, Xtest, y_train, y_test, dataset, task):
    labels = sorted(pd.unique(pd.concat([y_train, y_test]).astype(str)))
    n_classes = len(labels)
    if PROGRESS:   
        PROGRESS.update(stage="feature_selection", dataset=dataset, task=task, substage="feature selection methods", message=f"[feature_selection] {dataset}/{task}")
    for strategy in FS_METHODS:
        for model_name in ("RandomForest", "XGBoost", "LightGBM"):
            specif = spec(model_name, n_classes)
            if specif is None:
                continue
            k = key(kind="feature_selection", dataset=dataset, task=task, model=model_name, variant=strategy)
            if k in DONE:
                continue
            try:
                #selection is only on train and applied to test (no leakage)
                Xa, Xb, kept = select_features(X_train, Xtest, y_train, strategy)
                res, t, _ = fit_eval(specif.build(), Xa, y_train, Xb, y_test, labels)
                m = metrics(res)
                #feature list only <=60 otherwise too long for json
                record("feature_selection", {**m, "train_time_s": t, "n_features": len(kept), "n_features_orig": X_train.shape[1], "kept_features": kept if len(kept) <=60 else None}, dataset=dataset, task=task, model=model_name, variant=strategy)
                log.info("[B] %s/%s %s x %s: %d features f1m=%.4f (%.0fs)", dataset, task, model_name, strategy, len(kept), m["f1_macro"], t)
            except Exception as e:
                log.exception("[B] FAIL %s/%s %s x %s: %s", dataset, task, model_name, strategy, e)
            gc.collect()
            
'''Part c learning curve analysis (sample size)'''           
def part_c_sample_size(X_train, Xtest, y_train, y_test, dataset, task):
    labels = sorted(pd.unique(pd.concat([y_train, y_test]).astype(str)))
    n_classes = len(labels)
    if PROGRESS:  
        PROGRESS.update(stage="learning_curve", dataset=dataset, task=task, substage="training-set sizes", message=f"[learning_curve] {dataset}/{task}")
    for size in SAMPLE_SIZES:
        if size > len(X_train):
            continue #small dataset
        for model_name in LC_MODELS:
            specif = spec(model_name, n_classes)
            if specif is None:
                continue
            k = key(kind="learning_curve", dataset=dataset, task=task, model=model_name, variant=str(size))
            if k in DONE:
                continue
            try:
                Xa, ya = subsample(X_train, y_train, size)
                res, t, _ = fit_eval(specif.build(), Xa, ya, Xtest, y_test, labels) #same test set for all sizes
                m = metrics(res)
                record("learning_curve", {**m, "train_time_s": t, "sample_size": len(Xa)}, dataset=dataset, task=task, model=model_name, variant=str(size))
                log.info("[C] %s/%s %s x %d: f1m=%.4f (%.0fs)", dataset, task, model_name, size, m["f1_macro"], t)
            except Exception as e:
                log.exception("[C] FAIL %s/%s %s x %d: %s", dataset, task, model_name, size, e)
            gc.collect()
    prev = [r for r in SINK if r["kind"] == "learning_curve" and r["dataset"] == dataset and r["task"] == task]
    if prev:
        learning_curves([{"model": r["model"], "sample_size": r["sample_size"], "f1_macro": r["f1_macro"]} for r in prev], dataset, task, FIGURES_DIR / f"learning_curve_{dataset}_{task}.png")
        
'''Part d 5-fold stratified cross-validation analysis''' 
def part_d_cv(X_train, y_train, dataset, task, cap=CAP, cv_models=None):
    labels = sorted(pd.unique(y_train.astype(str)))
    n_classes = len(labels)
    if PROGRESS:  
        PROGRESS.update(stage="cv", dataset=dataset, task=task, substage="5-fold cross-validation", message=f"[cv] {dataset}/{task}")
    Xs, ys = subsample(X_train, y_train, cap) #smaller sampling
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    for model_name in (cv_models or CV_MODELS):
        specif = spec(model_name, n_classes)
        if specif is None:
            continue
        k = key(kind="cross_validation", dataset=dataset, task=task, model=model_name, variant="5fold")
        if k in DONE:
            #already completed, pull folds to SINK
            done_rows = [r for r in SINK if r["kind"] == "cv_fold" and r["dataset"] == dataset and r["task"] == task and r["model"] == model_name]
            fold_rows += [{"model": r["model"], "fold": r["variant"], "f1_macro": r["f1_macro"]} for r in done_rows]
            continue
        scores = []
        try:
            for i, (train_idx, test_idx) in enumerate(skf.split(Xs, ys)):
                res, t, _ = fit_eval(specif.build(), Xs.iloc[train_idx], ys.iloc[train_idx], Xs.iloc[test_idx], ys.iloc[test_idx], labels)
                m = metrics(res)
                scores.append(m["f1_macro"])
                #individual fold records for later plotting in SINK
                rec = {"kind": "cv_fold", "dataset": dataset, "task": task, "model": model_name, "variant": str(i), **META.get((dataset, task), {}), **m, "train_time_s": t}
                SINK.append(rec)
                fold_rows.append({"model": model_name, "fold": str(i), "f1_macro": m["f1_macro"]})
                save()
                if PROGRESS:   
                    PROGRESS.advance(model=model_name, substage=f"{model_name} - fold {i + 1} of 5")
            #summary record for the 5-fold cross-validation (mean ± std)
            record("cross_validation", {"f1_macro_mean": float(np.mean(scores)), "f1_macro_std": float(np.std(scores)), "folds": scores, "cv_rows": len(Xs)}, dataset=dataset, task=task, model=model_name, variant="5fold")
            log.info("[D] %s/%s %s: f1m=%.4f ± %.4f", dataset, task, model_name, np.mean(scores), np.std(scores))
        except Exception as e:
            log.exception("[D] FAIL %s/%s %s: %s", dataset, task, model_name, e)
        gc.collect()
    if fold_rows:
        cv_boxes(fold_rows, dataset, task, FIGURES_DIR / f"cv_folds_{dataset}_{task}.png")
        
'''extended study main entry point'''
def main():
    global PROGRESS   
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="*")
    ap.add_argument("--parts", nargs="*", default=["A", "B", "C", "D"])
    ap.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    ap.add_argument("--cv-cap", type=int, default=CAP, dest="cv_cap")
    ap.add_argument("--cv-models", nargs="*", default=CV_MODELS, dest="cv_models")
    args = ap.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    #resume from previous run if OUT_PATH exists
    if OUT_PATH.exists():
        try:
            SINK.extend(json.loads(OUT_PATH.read_text()))
            for r in SINK:
                if r["kind"] in ("imbalance", "feature_selection", "learning_curve", "cross_validation"):
                    DONE.add(key(kind=r["kind"], dataset=r["dataset"], task=r["task"], model=r["model"], variant=r["variant"]))
            log.info("resume: %d records loaded, %d done", len(SINK), len(DONE))
        except Exception:
            pass #corrupt file start fresh
    wanted = [d for d in (args.dataset or available_datasets()) if d in DATASETS]
    tasks = ("binary", "multiclass") if args.task == "both" else [args.task]
    log.info("Extended study started for datasets %s, tasks %s, parts %s", wanted, tasks, args.parts)
    #step count per task: 35 for A, 12 for B, 24 for C, 30 for D
    per_task = sum(n for part, n in (("A",35),("B",12),("C",24),("D",30)) if part in args.parts)
    PROGRESS = Progress(max(len(wanted) * len(tasks) * per_task, 1), phase="extended_study")
    PROGRESS.update(stage="starting", message=f"{len(wanted)} datasets x {len(tasks)} tasks")
    for name in wanted:
        PROGRESS.update(stage="loading", dataset=name, task="", model="", substage="reading and preprocessing the dataset", message=f"loading {name}")
        loader, _ = DATASETS[name]
        log.info("Processing dataset %s", name)
        df = loader()
        X, y, y_bin = build_xy(df)
        del df
        gc.collect()
        X_train, X_test, y_train, y_test, yb_train, yb_test = split(X, y, y_bin, dataset=name)
        X_trc, y_trc = subsample(X_train, y_train, ABLATION_ROWS) #shared training cap for all parts
        yb_trc = yb_train.loc[y_trc.index]  #align binary labels to same sample
        #same X, 2 targets: multiclass and binary 
        targets = {"multiclass": (X_trc, y_trc, X_test, y_test), "binary": (X_trc, yb_trc.map({0: "Benign", 1: "Attack"}), X_test, yb_test.map({0: "Benign", 1: "Attack"}))}
        #split_meta must describe the split, and y_trc is already capped to ABLATION_ROWS
        full_tr = {"multiclass": y_train, "binary": yb_train.map({0: "Benign", 1: "Attack"})}
        
        for task in tasks:  
            Xa, ya, Xb, yb = targets[task]
            META[(name, task)] = split_meta(name, full_tr[task], yb)
            if "A" in args.parts:
                part_a_imbalance(Xa, Xb, ya, yb, name, task)
            if "B" in args.parts:
                part_b_fs(Xa, Xb, ya, yb, name, task)
            if "C" in args.parts:
                part_c_sample_size(Xa, Xb, ya, yb, name, task)
            if "D" in args.parts:
                part_d_cv(Xa, ya, name, task, args.cv_cap, args.cv_models)
        del X, y, y_bin, X_train, X_test, X_trc
        gc.collect()
    log.info("Extended study completed. Results saved to %s", OUT_PATH)
    PROGRESS.finish(f"extended study complete: {len(SINK)} records") 
    
if __name__ == "__main__":
    main()
    
