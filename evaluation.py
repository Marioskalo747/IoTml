#Libraies
import sys
import time  #prediction time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,cohen_kappa_score,confusion_matrix,f1_score,log_loss,matthews_corrcoef,
    precision_recall_curve,precision_score,recall_score,roc_auc_score,roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sns.set_theme(style="whitegrid", font_scale=0.9) #constistent theme for all plots

'''positive class for binary metrics: "Attack" when present, otherwise the last label'''
def positive_label(labels):
    labels = list(labels)
    for cand in ("Attack", "attack", 1, "1"):
        if cand in labels:
            return cand
    return labels[-1] #fallback keeps the previous behaviour for non-IDS label sets

'''score for rank-based binary metrics (ROC-AUC, average precision)'''
def ranking_score(proba, ipos):
    p_pos = proba[:, ipos]
    p_neg = proba[:, 1 - ipos]
    if len(np.unique(p_neg)) > len(np.unique(p_pos)):
        return -p_neg #same ordering, more resolution
    return p_pos

'''Per-class breakdown of confusion matrix and metrics'''
def per_class_counts(y_true, y_pred, labels):       
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rows = []
    for i, label in enumerate(labels):
        tp = int(cm[i, i]) #diagonal: correct predictions
        fn = int(cm[i, :].sum() - tp) #false negatives
        fp = int(cm[:, i].sum() - tp) #false positives
        tn = int(total - tp - fn - fp) #true negatives
        if tp + fp:
            prec = tp / (tp + fp)
        else:
            prec = 0.0 #never predicted this class
        if tp + fn:
            rec = tp / (tp + fn)
        else:
            rec = 0.0 #never actual this class
        rows.append({"class": label, "TP": tp, "FP": fp, "TN": tn, "FN": fn,"support": int(cm[i, :].sum()),"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,"fpr": fp / (fp + tn) if fp + tn else 0.0,"specificity": tn / (tn + fp) if tn + fp else 0.0})
    return rows

'''central evaluation function'''
def evaluate_model(model, X_te, y_te, labels, *, proba=None):
    t0= time.perf_counter()
    y_pred = model.predict(X_te)
    pred_time = time.perf_counter() - t0  #prediction time
    res = {"accuracy": accuracy_score(y_te, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, y_pred), #mean recall
        "precision_macro": precision_score(y_te, y_pred, average="macro", labels=labels, zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro", labels=labels, zero_division=0),
        "precision_weighted": precision_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0),
        "recall_weighted": recall_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0),
        "f1_weighted": f1_score(y_te, y_pred, average="weighted", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y_te, y_pred), #strictest imbalanced metric [-1, 1]
        "cohen_kappa": cohen_kappa_score(y_te, y_pred), #aggreement above chance
        "predict_time_s": pred_time,
        "predict_us_per_sample": pred_time / max(len(y_te), 1) * 1e6, #real time use
        "per_class": per_class_counts(y_te, y_pred, labels)}
    #geometric mean of recalls
    recalls = [r["recall"] for r in res["per_class"] if r["support"] > 0]
    res["g_mean"] = float(np.prod(recalls) ** (1 / len(recalls))) if recalls else 0.0
    #metrics that require probabilities
    if proba is not None:
        try:
            if proba.ndim == 2 and proba.shape[1] > 2: 
                res["roc_auc_ovr"] = roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=labels)
                res["log_loss"] = log_loss(y_te, proba, labels=labels)
            else: #binary case
                #"Attack" is the positive class, the PR curve / average precision would describe the wrong class (PR is not symmetric)
                pos = positive_label(labels)
                ipos = list(labels).index(pos)
                if proba.ndim == 2 and ipos >= proba.shape[1]:
                    ipos = proba.shape[1] - 1 #labels wider than the probability matrix, keep the old behaviour
                    pos = list(labels)[ipos]
                p1 = proba[:, 1] if proba.ndim == 2 else proba #probability of labels[1], used for log_loss column order
                p = ranking_score(proba, ipos) if proba.ndim == 2 else (p1 if ipos == 1 else 1.0 - p1)
                y_pos = (np.asarray(y_te) == pos).astype(int)
                res["roc_auc_ovr"] = roc_auc_score(y_pos, p)
                res["average_precision_score"] = average_precision_score(y_pos, p)
                res["positive_label"] = str(pos)
                res["log_loss"] = log_loss(y_te, np.column_stack([1-p1, p1]), labels=labels)
        except Exception:
            res.setdefault("roc_auc_ovr", None) #missing class
    res["confusion_matrix"] = confusion_matrix(y_te, y_pred, labels=labels).tolist()
    res["labels"] = list(labels) #class order
    return res

'''ROC and PR curves for binary classification'''
def roc_pr_points(y_te, proba, labels, max_points=200):
    if proba is None or len(labels) != 2:
        return None #binary curves only
    if len(np.unique(np.asarray(y_te))) < 2:
        return None
    try:
        pos = positive_label(labels) #"Attack" is the positive class, not labels[1] ("Benign" when sorted)
        ipos = list(labels).index(pos)
        p1 = proba[:, 1] if proba.ndim == 2 else proba
        p = ranking_score(proba, ipos) if proba.ndim == 2 else (p1 if ipos == 1 else 1.0 - p1)
        y = (np.asarray(y_te) == pos).astype(int)
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        #Downsample to max_points for plotting  
        def ds(a):
            if len(a) <= max_points:
                return [float(v) for v in a]
            idx = np.linspace(0, len(a) - 1, max_points).astype(int)
            return [float(v) for v in np.asarray(a)[idx]]
        #compute on full dataset
        return {"fpr": ds(fpr), "tpr": ds(tpr),"precision": ds(precision),"recall": ds(recall),"auc": float(roc_auc_score(y, p)),"average_precision": float(average_precision_score(y, p)),"positive_label": str(pos)}
    except Exception:
        return None

'''confusion matrix'''
def plot_confusion_matrix(res: dict, title: str, path: Path):
    cm = np.array(res["confusion_matrix"], dtype=float) 
    norm = cm/cm.sum(axis=1, keepdims=True).clip(min=1) #normalized per true class (prevent division by zero)
    fig, ax = plt.subplots(figsize=(max(6, 0.78*len(res["labels"])),)*2) #square grows with the number of classes
    sns.heatmap(norm, annot=cm.astype(int), fmt="d", cmap="Blues", cbar=False, xticklabels=res["labels"], yticklabels=res["labels"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()  
    fig.savefig(path, dpi=300)
    plt.close(fig)

'''Bar chart 4 metrics comparison'''    
def model_comparison(df: pd.DataFrame, dataset: str, path: Path):
    metrics = ["accuracy", "f1_macro", "f1_weighted", "balanced_accuracy"]
    d = df.melt(id_vars="model", value_vars=metrics, var_name="metric",) #4 metrics as hue
    fig, ax = plt.subplots(figsize=(11,5))
    sns.barplot(data=d, x="model", y="value", hue="metric", ax=ax)
    ax.set_title(f"Model comparison {dataset}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

'''Class distribution on log axis bar chart'''
def class_distribution(y: pd.Series, dataset: str, path: Path, y_test: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    if y_test is None:
        counts = y.value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4878cf")
        ax.set_title(f"Class distribution {dataset}")
    else:
        tr = y.value_counts()
        te = y_test.value_counts()
        order = (tr.add(te, fill_value=0)).sort_values(ascending=False).index
        d = pd.DataFrame({"class": list(order) * 2,
                          "split": ["train"] * len(order) + ["test"] * len(order),
                          "flows": [float(tr.get(c, 0)) for c in order] + [float(te.get(c, 0)) for c in order]})
        #a class with 0 rows on one side must stay visible (it is floored at 0.5)
        d["flows"] = d["flows"].clip(lower=0.5)
        sns.barplot(data=d, x="class", y="flows", hue="split", ax=ax,
                    palette={"train": "#4878cf", "test": "#d65f5f"})
        ax.set_title(f"Class distribution {dataset} (after dedup and split; 0.5 = class absent)")
    ax.set_ylabel("flows")
    ax.set_yscale("log")  #log scale for imbalanced datasets
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

'''Feature importance bar chart'''    
def feature_importance(names, importances, dataset: str, model: str, path: Path, top=20):
    imp = np.asarray(importances, dtype=float)
    imp = imp / imp.sum() if imp.sum() else imp #normalize to sum to 1
    order = np.argsort(imp)[::-1][:top]  #descending order, top features
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp[order], y=np.array(names)[order], ax=ax, color="#d65f5f")
    ax.set_title(f"Top features {model} on {dataset}")
    ax.set_xlabel("Relative importance (sums to 1)")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
'''ROC curves order by AUC'''    
def roc_curves(curves: dict[str, dict], dataset: str, path: Path):
    fig, ax = plt.subplots(figsize=(7,6)) 
    for model, c in sorted(curves.items(), key=lambda kv: -kv[1]["auc"]): #best models first
        ax.plot(c["fpr"], c["tpr"], label=f"{model} (AUC={c['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5) #diagonal line for random classifier
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves {dataset} (binary)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

'''Precision-Recall curves'''    
def pr_curves(curves: dict[str, dict], dataset: str, path: Path):
    fig, ax = plt.subplots(figsize=(7,6))
    for model, c in sorted(curves.items(), key=lambda kv: -kv[1]["average_precision"]):
        ax.plot(c["recall"], c["precision"], lw=1.8, label=f"{model} (AP={c['average_precision']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves {dataset} (binary)")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
'''Generic metric heatmap'''   
def metric_heatmap(df: pd.DataFrame, metric: str, title: str, path: Path, fmt=".3f"):
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * df.shape[1], 1.2 + 0.45 * df.shape[0])) 
    sns.heatmap(df, annot=True, fmt=fmt, cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": metric})
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
'''Learning curves as function of training sample size'''   
def learning_curves(rows: list[dict], dataset: str, task:str, path: Path, metric="f1_macro"):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, g in df.groupby("model"):
        g = g.sort_values("sample_size") #correct order for line plot
        ax.plot(g["sample_size"], g[metric], marker="o", label=model)
    ax.set_xscale("log") #size scale
    ax.set_xlabel("Training samples")
    ax.set_ylabel(metric)
    ax.set_title(f"Learning curves {dataset} ({task})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3) #grid lines for log scale
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
    
'''Imbalance strategies comparison'''
def strategy_comparison(rows: list[dict], dataset: str, task: str, path: Path, metric="f1_macro"):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df, x="model", y=metric, hue="imbalance", ax=ax)
    ax.set_title(f"Imbalance Strategies {dataset} ({task})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
'''Default vs optuna-tuned hyperparameter comparison'''    
def tuning_improvement(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    df["combo"]= df["dataset"] + "\n" + df["model"]
    d = df.melt(id_vars="combo", value_vars=["f1_default", "f1_tuned"], var_name="config", value_name="f1_macro")
    d["config"] = d["config"].map({"f1_default": "F1 Default", "f1_tuned": "Optuna-Tuned"})
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * df["combo"].nunique()), 5))
    sns.barplot(data=d, x="combo", y="f1_macro", hue="config", ax=ax)
    lo = max(0.0, d["f1_macro"].min() - 0.05)
    ax.set_ylim(lo, 1.02)
    ax.set_xlabel("")
    _delta = df["f1_tuned"].mean() - df["f1_default"].mean()
    _n_up = int((df["f1_tuned"] > df["f1_default"]).sum())
    ax.set_title(f"Optuna tuning vs defaults (Macro-F1): mean change {_delta:+.4f}, "
                 f"{_n_up} of {len(df)} combinations improved")
    plt.xticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
    
'''Optuna hyperparameter tuning history'''    
def optuna_history(trials: list[dict], dataset: str, model: str, path: Path):
    df = pd.DataFrame(trials)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.scatter(df["number"], df["value"], s=22, alpha=0.6, label="trial")
    ax.plot(df["number"], df["value"].cummax(), color="#d65f5f", lw=2, label="best so far" )
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Cross Validation Macro-F1")
    ax.set_title(f"Optuna hyperparameter tuning history {model} on {dataset}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
'''Per class F1 score bar chart ordered by frequency'''   
def class_f1(per_class: list[dict], dataset: str, model: str, task: str, path: Path):
    df = pd.DataFrame(per_class).sort_values("support", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["class"], df["f1"], color="#4878cf")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"{c}" + chr(10) + f"n={s:,}" for c, s in zip(df["class"], df["support"])])
    ax.set_title(f"Per-class F1 scores {model} on {dataset} ({task})")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
'''Boxplot of 5 cross-validation folds for variability analysis'''    
def cv_boxes(rows: list[dict], dataset: str, task: str, path: Path):
    df= pd.DataFrame(rows)
    order = (df.groupby("model")["f1_macro"].mean().sort_values(ascending=False).index) #best models on left
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="model", y="f1_macro", order=order, ax=ax, color="#7fbf7f")
    sns.stripplot(data=df, x="model", y="f1_macro", order=order, ax=ax, color="black", size=3, alpha=0.6)
    ax.set_title(f"5-fold stratified CV {dataset} ({task})")
    ax.set_xlabel("")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    
