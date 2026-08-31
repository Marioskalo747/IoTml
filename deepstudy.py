#Libraries
import gc
import json
import sys
import time
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg") #png files only no GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_sample_weight #per sample weights for imbalanced datasets
 
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIGURES_DIR, RESULTS_DIR, RANDOM_STATE
from loaders import DATASETS, available_datasets
from preprocessing import build_xy, make_preprocessor, split, split_meta
from progress import Progress

sns.set_theme()
ABLATION_MODELS = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "MLP"] 
STRATEGIES = ["none", "class_weight", "oversample"] #3 ways to handle imbalance: none, class_weight, oversample
ABLATION_TRAIN_CAP = 120000 #training cap
MLP_CAP = 60000 #tighter cap for MLP 
CURVE_SIZES = [2000, 5000, 10000, 25000, 50000, 100000, 250000] #log-spced points for learning curves

'''MOdel building with imbalance strategies'''
def build_model(name: str, strategy: str, n_classes: int):
    cw = strategy == "class_weight"
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=25, min_samples_leaf=3, class_weight="balanced" if cw else None, random_state=RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced" if cw else None, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "XGBoost":
        if cw:
            return None #XGBoost has no class_weight parameter, use sample weights instead
        return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.15, subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss" if n_classes > 2 else "logloss", random_state=RANDOM_STATE, n_jobs=-1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, num_leaves=64, learning_rate=0.1, reg_lambda=1.0, class_weight="balanced" if cw else None, verbose=-1, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "CatBoost":
        #CatBoost has auto_class_weights="Balanced" parameter
        return CatBoostClassifier(iterations=300, depth=8, learning_rate=0.15, auto_class_weights="Balanced" if cw else None, verbose=0, random_seed=RANDOM_STATE, allow_writing_files=False)
    if name == "MLP":
        if cw:
            return None #sklearn MLPClassifier has no class_weight parameter
        return MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=512, max_iter=300, early_stopping=True, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model name: {name}")

'''XGBoost with balancing in samplwe weights'''
def xgb_with_weights(n_classes):
    return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.15, subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss" if n_classes > 2 else "logloss", random_state=RANDOM_STATE, n_jobs=-1)

'''Capped oversampling to avoid excessive duplication of minority classes'''
def capped_oversample(X, y, factor=3.0):
    counts = pd.Series(y).value_counts()
    #min avoids 10x data explosion
    target_n = int(min(counts.max(), factor*counts.mean()))\
    #max not allows to RandomOverSampler to shrink any class
    strategy = {c: max(int(n), min(target_n, int(counts.max()))) for c, n in counts.items()}
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=RANDOM_STATE)
    return ros.fit_resample(X, y)

'''Stratified sample of n rows'''
def stratified_take(X, y, n):
    if len(X) <= n:
        return X, y
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False).apply(lambda g: g.sample(n=max(1, round(len(g)*n/len(X))), random_state=RANDOM_STATE)))
    return X.loc[idx.index], y.loc[idx.index]

'''Lightweight evaluation with 5 metrics.'''
def evaluate_model(clf_pipe, X_te, y_te, labels=None):
    y_pred = clf_pipe.predict(X_te)
    return{"accuracy": accuracy_score(y_te, y_pred), "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),"f1_macro": f1_score(y_te, y_pred, average="macro", labels=labels, zero_division=0), 
    "f1_weighted": f1_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0),"mcc": matthews_corrcoef(y_te, y_pred)}

'''Persists results'''
def save(out):
    tmp = RESULTS_DIR / "deep_study.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, default=str)
    for _try in range(5):  
        try:                
            tmp.replace(RESULTS_DIR / "deep_study.json") #atomic replace to avoid partial writes
            break           
        except PermissionError:
            time.sleep(0.3) #avoid file lock issues on Windows

'''Figures and CSV'''
def make_figures(out):
    abl = pd.DataFrame(out["ablation"])
    if not abl.empty:
        abl.to_csv(RESULTS_DIR / "table_ablation.csv", index=False)
        for ds, grp in abl.groupby("dataset"):
            #models: x, strategy: hue
            fig, ax = plt.subplots(figsize=(9, 4.5))
            sns.barplot(data=grp, x="model", y="f1_macro", hue="strategy", ax=ax)
            ax.set_title(f"Ablation study: {ds}")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("")
            ax.set_ylabel("F1 macro score")
            plt.xticks(rotation=15)
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"ablation_{ds}.png", dpi=300)
            plt.close(fig)
            
    lc = pd.DataFrame(out["learning_curves"])
    if not lc.empty:
        lc.to_csv(RESULTS_DIR / "table_learning_curves.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for ax, model in zip(ax, ["RandomForest", "XGBoost"]):
            d = lc[lc.model == model]
            sns.lineplot(data=d, x="n", y="f1_macro", hue="dataset", marker="o", ax=ax)
            ax.set_xscale("log")  #log scale
            ax.set_title(f"Learning curve: {model}")
            ax.set_xlabel("Training rows (log scale)")
            ax.set_ylabel("F1 macro score")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=300)
        plt.close(fig)
        
    if out.get("tuning"):
        t = out["tuning"] #default and tuned metrics
        pd.DataFrame([{"config": "default", **t["default"]},{"config": "tuned", **{k: v for k, v in t["tuned"].items() if k not in ("best_params",)}}]).to_csv(RESULTS_DIR / "table_tuning.csv", index=False)
        
'''Tuning experiment with RandomizedSearchCV'''   
def tuning_experiment():
    loader, _ = DATASETS["ciciot2023"] #largest dataset for representative case
    df = loader()
    X, y, y_bin = build_xy(df)
    del df
    X_tr, X_te, y_tr, y_te, _, _ = split(X, y, y_bin, dataset="ciciot2023")
    le = LabelEncoder().fit(pd.concat([y_tr, y_te]).astype(str)) #one encoder for all classes
    yt = pd.Series(le.transform(y_tr.astype(str)), index=X_tr.index) #index is kept for stratified_take
    yv = le.transform(y_te.astype(str))
    Xs, ys = stratified_take(X_tr, yt, 60000) #searh on a sample
    #baseline (default hyperparameters)
    base = Pipeline([("pre", make_preprocessor(Xs)), ("clf", XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.15, subsample=0.9, colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1))])
    t0 = time.perf_counter()
    base.fit(Xs, ys)
    base_time = time.perf_counter() - t0
    all_labels = list(range(len(le.classes_)))
    base_m = evaluate_model(base, X_te, yv, labels=all_labels)
    #20 configurations at random from grid
    search_pipe = Pipeline([("pre", make_preprocessor(Xs)), ("clf", XGBClassifier(tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1))])
    #clf targets the pipeline step so preprocessor refitted per fold
    space = {"clf__n_estimators": [100, 200, 300, 400, 500],"clf__max_depth": [4, 6, 8, 10, 12],"clf__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2, 0.3],
        "clf__subsample": [0.6, 0.75, 0.9, 1.0],"clf__colsample_bytree": [0.6, 0.75, 0.9, 1.0],"clf__min_child_weight": [1, 2, 4, 8]}
    rs = RandomizedSearchCV(search_pipe, space, n_iter=20, scoring="f1_macro", cv=3, verbose=1, random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.perf_counter()
    rs.fit(Xs, ys)
    search_time = time.perf_counter() - t0
    tuned_m = evaluate_model(rs.best_estimator_, X_te, yv, labels=all_labels)
    res = {"dataset": "ciciot2023", "model": "XGBoost", "n_iter": 20, "cv": 3, "default": {**base_m, "train_time_s": base_time}, "tuned": {**tuned_m, "search_time_s": search_time, "best_params": {k.replace("clf__", ""): v for k, v in rs.best_params_.items()},"cv_best_f1": rs.best_score_},"delta_f1": tuned_m["f1_macro"] - base_m["f1_macro"]}
    print(f"[tuning] Default: f1m={base_m['f1_macro']:.4f}" f"Tuned: f1m={tuned_m['f1_macro']:.4f} (Δ={res['delta_f1']:+.4f}, search_time {search_time/60:.0f}min)")
    return res

'''Imbalance ablation study, learning curves, and tuning experiment'''    
def main():
    out = {"ablation": [], "learning_curves": [],"tuning_results": []}
    datasets = [d for d in available_datasets() if d in DATASETS]
    n_abl = len(datasets) * len(ABLATION_MODELS) * len(STRATEGIES)
    n_curve = len(datasets) * len(CURVE_SIZES)*2 #2 models for learning curves
    prog = Progress(n_abl + n_curve + 1, phase="deep_study")   #1 for tuning experiment
    prog.update(stage="starting", message=f"Deep study: {datasets}")
    
    for ds in datasets:
        prog.update(stage="loading", dataset=ds,message=f"Loading dataset {ds}")
        loader, _ = DATASETS[ds]
        df = loader()
        X, y, y_bin = build_xy(df)
        del df
        gc.collect()
        #dataset=ds: without it every call lands under the key "unnamed" in split_audit.json
        X_tr, X_te, y_tr, y_te, _, _ = split(X, y, y_bin, dataset=ds)
        meta = split_meta(ds, y_tr, y_te) #stamped on every ablation and learning-curve row below
        #XGBoost requires integer labels
        le = LabelEncoder().fit(pd.concat([y_tr, y_te]).astype(str))
        yt, yv = le.transform(y_tr.astype(str)), le.transform(y_te.astype(str))
        yt = pd.Series(yt, index=X_tr.index) #one index for stratified_take
        n_classes = len(le.classes_)
        all_labels = list(range(n_classes)) #full vocabulary for every macro average below
        #ablation study (imbalanced strategies)
        for model_name in ABLATION_MODELS:
            for strategy in STRATEGIES:
                tag = f"{ds}/{model_name}/{strategy}"
                prog.update(stage="ablation", dataset=ds,model=model_name,task=strategy,substage=f"imbalance strategy: {strategy}",message=f"[{tag}] Training and evaluating model")
                try:
                    cap = MLP_CAP if model_name == "MLP" else ABLATION_TRAIN_CAP
                    Xa, ya = stratified_take(X_tr, yt, cap)
                    sample_weight = None
                    if strategy == "oversample":
                        Xa, ya = capped_oversample(Xa, ya) #oversampling
                    clf = build_model(model_name, strategy, n_classes)
                    if clf is None and model_name == "XGBoost":
                        #per sample weights for XGBoost with class_weight strategy
                        clf = xgb_with_weights(n_classes)
                        sample_weight = compute_sample_weight("balanced", ya)
                    if clf is None:
                        prog.advance()
                        continue #MPL has no class_weight
                    pipe = Pipeline([("pre", make_preprocessor(Xa)), ("clf", clf)])
                    t0 = time.perf_counter()
                    if sample_weight is not None:
                        pipe.fit(Xa, ya, clf__sample_weight=sample_weight) #clf step 
                    else:
                        pipe.fit(Xa, ya)
                    dt = time.perf_counter() - t0
                    m = evaluate_model(pipe, X_te, yv, labels=all_labels)
                    m.update({"dataset": ds, "task": "multiclass", "model": model_name, "strategy": strategy,
                              "train_rows": len(Xa), "train_time_s": dt, **meta})
                    out["ablation"].append(m)
                    print(f"[abl] {tag:45s} f1m={m['f1_macro']:.4f} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} mcc={m['mcc']:.4f} time={dt:.2f}s")
                except Exception as e:
                    print(f"[{tag}] Error: {e}") #fail
                finally:
                    prog.advance()
                    save(out) #checkpoint
                    gc.collect()
        #learning curves 
        for model_name in ["RandomForest", "XGBoost"]:
            for size in CURVE_SIZES:
                prog.update(stage="learning_curve", dataset=ds,model=model_name, substage=f"{size:,} training rows", message=f"[{ds}/{model_name}] Learning curve with {size} training rows")
                try:
                    if size > len(X_tr):
                        prog.advance()
                        continue #small dataset, skip
                    Xc, yc = stratified_take(X_tr, yt, size)
                    clf = build_model(model_name, "none", n_classes) #no strategy
                    pipe = Pipeline([("pre", make_preprocessor(Xc)), ("clf", clf)])
                    t0 = time.perf_counter()
                    pipe.fit(Xc, yc)
                    dt = time.perf_counter() - t0
                    m = evaluate_model(pipe, X_te, yv, labels=all_labels) #same test size set for comparison
                    out["learning_curves"].append({"dataset": ds, "task": "multiclass", "model": model_name, "n": int(len(Xc)),
                                                   "f1_macro": m["f1_macro"], "accuracy": m["accuracy"], "mcc": m["mcc"],
                                                   "train_time_s": dt, **meta})
                    print(f"[curve] {ds}/{model_name:10s} rows={len(Xc)}")             
                except Exception as e:
                    print(f"[{ds}/{model_name}] Error: {e}")
                finally:
                    prog.advance()
                    save(out)
        del X_tr, X_te, X, y
        gc.collect()
    #Hyperparameter tuning experiment    
    try:
        prog.update(stage="tuning", dataset="ciciot2023", model="XGBoost", substage="RandomizedSearchCV, 20 candidates x 3 folds", message="RandomizedSearchCV")
        out["tuning"] = tuning_experiment()
    except Exception as e:
        print(f"[tuning] Error: {e}")
    prog.advance()
    save(out)
    make_figures(out)
    prog.finish("Deep study completed")
    print("deep study completed.", len(out["ablation"]), "ablation,", len(out["learning_curves"]), "curve points")
    
    
if __name__ == "__main__":
    main()
    