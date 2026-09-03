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
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from evaluation import evaluate_model, optuna_history, tuning_improvement
from progress import Progress
from loaders import DATASETS, available_datasets
from models_zoo import LabelEncodedClassifier, model_zoo
from preprocessing import build_xy, make_preprocessor, split, split_meta

optuna.logging.set_verbosity(optuna.logging.WARNING)


TUNE_ROWS = 30000 #25 trials x 3 folds x 30k rows
N_TRIALS = 25 #time/quality balance for TPE
CV_FOLDS = 3 
OUT_PATH = RESULTS_DIR / "tuning_results.json"
#logs
log = logging.getLogger("tuning")
logging.basicConfig( level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler(RESULTS_DIR / "tuning.log",encoding="utf-8")])

'''XGBoost (tree depth, learning rate, regularization)'''
def xgb(trial, n_classes):
    p= {"n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), #multiplicative scale
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0), #stochastic gradient boosting
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0), #minimum gain to split a node
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    return LabelEncodedClassifier(XGBClassifier(**p, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE, eval_metric="mlogloss" if n_classes>2 else "logloss"))

'''LightGBM (num leaves, learning rate, regularization)'''
def lgbm(trial, n_classes):
    p = {"n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True), #multiplicative scale
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    #subsample_freq=1 
    return LGBMClassifier(**p, subsample_freq=1 , n_jobs=-1, verbose=-1, random_state=RANDOM_STATE)

'''CatBoost (iterations, learning rate, depth, regularization)'''
def cat(trial, n_classes):
    p = {"iterations": trial.suggest_int("iterations", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255)} #numeric discresation resolution
    return CatBoostClassifier(**p, verbose=0, random_state=RANDOM_STATE, allow_writing_files=False)

'''Random Forest (fewer knobs, max features controls tree diversity)'''
def rf(trial, n_classes):
    p = {"n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 8, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])}
    return RandomForestClassifier(**p, n_jobs=-1, random_state=RANDOM_STATE)

#4 stringest get tuned
SEARCH_SPACE = {"XGBoost": xgb, "LightGBM": lgbm, "CatBoost": cat, "RandomForest": rf}

'''Return same model with default hyperparameters (for comparison)'''
def default_model(name: str, n_classes: int):
    for spec in model_zoo(n_classes):
        if spec.name == name:
            return spec.build()
    raise KeyError(f"Model {name} not found in zoo for {n_classes} classes")

'''Stratified sampling for search phase '''
def subsample(X, y, cap):
    if len(X) <= cap:
        return X, y
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False).apply(lambda x: x.sample(n=max(CV_FOLDS, round(len(x)*cap/len(X))), random_state=RANDOM_STATE)))
    return X.loc[idx.index], y.loc[idx.index]

PROGRESS = None #optuna callback has no extra args

'''Tuning (optuna search, refit best config, comparison)'''
def tune_one(model_name, X_train, X_test, y_train, y_test, dataset, task, n_trials, tune_rows=TUNE_ROWS):
    labels = sorted(pd.unique(pd.concat([y_train, y_test]).astype(str)))
    n_classes = len(labels)
    pre = make_preprocessor(X_train) #final processor
    #the search runs on a subsample but the winner is refitted on the full training split
    Xs, ys = subsample(X_train, y_train, tune_rows) #small sample
    search_fraction = len(Xs) / max(len(X_train), 1)
    log.info("[T] %s/%s: tuning %s on %d of %d training rows (%.1f%%, %d classes)", dataset, task, model_name, len(Xs), len(X_train), 100 * search_fraction, n_classes)
    if search_fraction < 0.5:
        log.warning("[T] %s/%s/%s: hyperparameters searched on %.1f%% of the training rows they are refitted on", dataset, task, model_name, 100 * search_fraction)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    '''mean f1 macro over 3 folds (optuna objective)'''
    def objective(trial):
        clf = SEARCH_SPACE[model_name](trial, n_classes) #pipeline (no data leakage)
        pipe = Pipeline([("preprocessor", make_preprocessor(Xs)), ("clf", clf)])
        scores = cross_val_score(pipe, Xs, ys.astype(str), cv=skf, scoring="f1_macro", n_jobs=1) #parallelization
        return float(scores.mean())
    
    if PROGRESS: 
        PROGRESS.update(stage="tuning", dataset=dataset, task=task, model=model_name,substage=f"trial 0 of {n_trials}", message=f"tuning {model_name} on {dataset} {task}",
                        details={"tuning": {"dataset": dataset, "task": task, "model": model_name,"trials": n_trials, "cv_folds": CV_FOLDS, "tune_rows": len(Xs)}})
    
    '''callback after each trial'''
    def on_trial(study_, trial_):  
        if not PROGRESS:
            return
        try:
            best = study_.best_value
        except ValueError:
            best = float("nan") #no trials completed yet
        PROGRESS.advance(substage=f"trial {trial_.number + 1} of {n_trials} - best f1_macro {best:.4f}", details={"tuning": {"trial": trial_.number + 1, "best_cv_f1_macro": best}})
    #seed sampler for reproducibility, 8 random trials to start
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE, n_startup_trials=8))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, gc_after_trial= True, callbacks=[on_trial])   
    tune_time = time.perf_counter() - t0
    
    if PROGRESS:   
        PROGRESS.update(stage="refit", substage=f"retraining the best of {n_trials} trials on the full training split")
    #rebuilds model with best hyperparameters on full training split
    best_clf = SEARCH_SPACE[model_name](optuna.trial.FixedTrial(study.best_params), n_classes)
    pipe = Pipeline([("preprocessor", pre), ("clf", best_clf)])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train.astype(str)) #final fit
    train_time = time.perf_counter() - t0
    proba = None
    try:
        proba = pipe.predict_proba(X_test)
    except Exception:
        pass
    res = evaluate_model(pipe, X_test, y_test.astype(str), labels, proba=proba)
    #comparison with default hyperparameters
    dflt = Pipeline([("preprocessor", make_preprocessor(X_train)), ("clf", default_model(model_name, n_classes))]).fit(X_train, y_train.astype(str))
    res_d=evaluate_model(dflt, X_test, y_test.astype(str), labels)
    tag = f"{dataset}__{task}__{model_name}__tuned"
    joblib.dump(pipe, MODELS_DIR / f"{tag}.joblib", compress=3) #compare size/speed tradeoff
    trials = [{"number": t.number, "value": t.value} for t in study.trials if t.value is not None]
    optuna_history(trials, dataset, model_name, FIGURES_DIR / f"optuna_{dataset}_{task}_{model_name}.png") #convergence plot
    meta = split_meta(dataset, y_train, y_test)
    out = {"dataset": dataset, "task": task, "model": model_name,
            "n_trials": n_trials, "cv_folds": CV_FOLDS, "tune_rows": len(Xs),
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "train_rows_available": int(len(X_train)), "capped": False, "train_cap": None,
            #same two columns run_all records computed the same way
            "n_features": int(pipe.named_steps["preprocessor"].transform(X_test.head(1)).shape[1]),
            "train_min_class_rows": int(y_train.astype(str).value_counts().min()),
            **meta,
            #search vs refit size difference
            "refit_rows": int(len(X_train)), "search_fraction": float(search_fraction),
            "best_params": study.best_params,
            "best_cv_f1_macro": study.best_value,
            "tune_time_s": tune_time, "train_time_s": train_time,
            #predict_us_per_sample belongs here
            "tuned": {k: res[k] for k in("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted","precision_macro", "recall_macro", "mcc", "cohen_kappa","g_mean", "predict_us_per_sample")},
            "default": {k: res_d[k] for k in("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted","precision_macro", "recall_macro", "mcc", "cohen_kappa","g_mean")},
            "tuned_roc_auc": res.get("roc_auc_ovr"),
            "trials": trials,
            "per_class_tuned": res["per_class"],
            "confusion_matrix_tuned": res["confusion_matrix"],
            "labels": labels}
    
    log.info("done tuning %s/%s %s: best_cv_f1_macro=%.4f, tuned f1_macro=%.4f, default f1_macro=%.4f",dataset, task, model_name, study.best_value, res["f1_macro"], res_d["f1_macro"])
    return out

'''Tunes datasets/models/tasks with results in all_results.json'''
def main():
    global PROGRESS
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="dataset names (default: all)")
    ap.add_argument("--task", choices=["binary", "multiclass", "both"], default="both")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--models", nargs="*", default=list(SEARCH_SPACE))
    #raise to close the gap between the search sample and the full refit training split
    ap.add_argument("--tune-rows", type=int, default=TUNE_ROWS, dest="tune_rows")
    args = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    #resume if existing results file 
    existing: list[dict] = []
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
        except Exception:
            existing = []
    done = {(r["dataset"], r["task"], r["model"]) for r in existing}
    
    wanted = [d for d in (args.datasets or available_datasets()) if d in DATASETS]
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    log.info("Tuning datasets=%s tasks=%s models=%s trials=%d", wanted, tasks, args.models, args.trials)
    remaining = [(n, t, m) for n in wanted for t in tasks for m in args.models if (n, t, m) not in done]
    #progress
    PROGRESS = Progress(max(len(remaining) * args.trials, 1), phase="tuning")
    PROGRESS.update(stage="starting", message=f"{len(remaining)} combinations x {args.trials} trials")
    for name in wanted:
        todo = [(t, m) for t in tasks for m in args.models if (name, t ,m) not in done]
        if not todo:
            log.info("Skipping %s: already tuned for all tasks/models", name)
            continue #dataset already tuned for all tasks/models
        PROGRESS.update(stage="loading", dataset=name, task="", model="",substage="reading and preprocessing the dataset", message=f"loading {name}")
        loader, _ = DATASETS[name]
        df = loader()
        X, y, y_bin = build_xy(df)
        del df
        gc.collect()
        X_train, X_test, y_train, y_test, yb_train, yb_test = split(X, y, y_bin, dataset=name)
        #multiclass and binary targets for tuning
        targets = {"multiclass": (y_train, y_test), "binary": (yb_train.map({0: "Benign", 1: "Attack"}), yb_test.map({0: "Benign", 1: "Attack"}))}
        for t, m in todo:
            try:
                res = tune_one(m, X_train, X_test, targets[t][0], targets[t][1], name, t, args.trials, args.tune_rows)
                existing.append(res)
                OUT_PATH.write_text(json.dumps(existing, indent=1, default=str)) #checkpoint after each combination
            except Exception as e:
                log.exception("FAIL %s/%s %s: %s", name, t, m, e)
            gc.collect()
        del X, y, y_bin, X_train, X_test
        gc.collect()
    #figures tuning/combination improvement        
    rows = [{"dataset": r["dataset"] + ("/bin" if r["task"]=="binary" else "/multi"), "model": r["model"], "f1_default": r["default"]["f1_macro"], "f1_tuned": r["tuned"]["f1_macro"]} for r in existing]
    if rows:
        tuning_improvement(rows, FIGURES_DIR / "tuning_improvement.png")
    #merge in central results
    allp = RESULTS_DIR / "all_results.json"
    if allp.exists():
        all = json.loads(allp.read_text(encoding="utf-8"))
        known = {(r["dataset"], r["task"], r["model"]) for r in all}
        #family comes from the base model, not hardcoded: RandomForest__tuned is "ensemble", not "boosting"
        families = {s.name: s.family for s in model_zoo(3)}
        added = 0
        for r in existing:
            _name = r["model"] + "__tuned"
            if (r["dataset"], r["task"], _name) in known:
                continue #already in all_results.json
            all.append({**r["tuned"], "dataset": r["dataset"], "task": r["task"],"model": _name, "family": families.get(r["model"], "boosting"), "params": r["best_params"],"roc_auc_ovr": r.get("tuned_roc_auc"),
                        "per_class": r["per_class_tuned"],"confusion_matrix": r["confusion_matrix_tuned"], "labels": r["labels"], "train_time_s": r.get("train_time_s"),
                        #sizes and split metadata so the tuned rows are not blank in the overview
                        **{k: r.get(k) for k in ("train_rows", "test_rows", "train_rows_available", "capped", "train_cap", "split_mode", "evaluable",
                                                 "n_test_classes", "classes_missing_from_train", "min_train_per_class", "classes_undertrained",
                                                 "fully_learnable", "n_features", "train_min_class_rows")}})
            added += 1
        allp.write_text(json.dumps(all, indent=1, default=str), encoding="utf-8")
        log.info("added %d tuned models to all_results.json", added)
    log.info("Tuning completed. Results saved to %s (%d records)", OUT_PATH, len(existing))
    PROGRESS.finish(f"tuning complete: {len(existing)} combinations")
    
        
if __name__ == "__main__":
    main()