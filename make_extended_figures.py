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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import Progress
from config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from evaluation import (metric_heatmap, class_f1, pr_curves, roc_curves, roc_pr_points)
from loaders import DATASETS, available_datasets
from preprocessing import build_xy, split
#logs
log = logging.getLogger("figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "figures.log", encoding="utf-8")])

#strong models for ROC/PR curves
TOP_ROC_MODELS = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "TorchMLP", "LogisticRegression"]

'''Fill in missing metrics from older results from confusion matrix'''
def backfill_metrics():
    path = RESULTS_DIR / "all_results.json"
    results = json.loads(path.read_text())
    n_fix = 0
    for r in results:
        cm = np.array(r.get("confusion_matrix", []), dtype=float)
        if cm.size == 0:
            continue #no confusion matrix, cannot backfill
        n = cm.sum()
        if "cohen_kappa" not in r:
            #kappa = (observed - expected agreement) / (1-expected agreement)
            po = np.trace(cm) / n #correct predictions
            pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n) #expected agreement by chance
            r["cohen_kappa"] = (po - pe) / (1 - pe) if pe < 1 else 0.0 #pe=1 undefined, set to 0.0
            n_fix += 1
        recalls = [row[i] / row.sum() for i, row in enumerate(cm) if row.sum()]
        r.setdefault("g_mean", float(np.prod(recalls) ** (1 / len(recalls))) if recalls and min(recalls) > 0 else 0.0)
        #f1 from precision and recall, specificity from TN and FP
        for i, pc in enumerate(r.get("per_class", [])):
            if "f1" not in pc:
                p = pc["precision"]
                q = pc["recall"]
                pc["f1"] = 2 * p * q / (p + q) if p + q else 0.0
            if "specificity" not in pc:
                tn = pc["TN"]
                fp = pc["FP"]
                pc["specificity"] = tn / (tn + fp) if tn + fp else 0.0
    path.write_text(json.dumps(results, indent=1, default=str))
    log.info("Backfilled metrics for %d results %d", len(results), n_fix)
    return results

'''ROC and PR curves for binary classification datasets'''
def roc_pr_for_datasets(name):
    loader, _ = DATASETS[name]
    df = loader()
    X, y, y_bin = build_xy(df)
    del df
    gc.collect() #heavy datasets may cause memory issues
    #splits as training 
    _, X_te, _, _, _, yb_te = split(X, y, y_bin, dataset=name)
    y_te =  yb_te.map({0: "Benign", 1: "Attack"})
    labels = sorted(set(y_te))
    curves = {}
    for model in TOP_ROC_MODELS:
        p = MODELS_DIR / f"{name}__binary__{model}.joblib" #name__task__model.joblib
        if not p.exists():
            continue #model not trained for this dataset
        try:
            pipe = joblib.load(p)
            proba = pipe.predict_proba(X_te)
            classes = list(getattr(pipe, "classes_", labels))
            pos = classes.index("Attack") if "Attack" in classes else 1 #positive class index
            #reorder columns to [negative, positive] for ROC/PR
            pts = roc_pr_points(y_te, proba[:, [1-pos,pos]], ["Benign", "Attack"])
            if pts:
                curves[model] = pts
            log.info("ROC/PR curves for %s/%s: auc=%.4f", name, model, pts["auc"] if pts else float("nan"))
        except Exception as e:
            log.warning("Failed to compute ROC/PR for %s/%s: %s", name, model, e)
        gc.collect()
    if curves:
        roc_curves(curves, name, FIGURES_DIR / f"roc_{name}.png")
        pr_curves(curves, name, FIGURES_DIR / f"pr_{name}.png")
        (RESULTS_DIR / f"roc_pr_{name}.json").write_text(json.dumps(curves, indent=1)) #save curves for later use
    del X, X_te
    gc.collect()
    return curves

'''Model/dataset heatmaps (no LODO)'''
def heatmaps(results):
    df = pd.DataFrame([{k: r.get(k) for k in ("dataset", "task", "model", "f1_macro", "accuracy", "mcc", "cohen_kappa", "balanced_accuracy")} for r in results if not str(r["dataset"]).startswith("LODO")])
    for task in ("binary", "multiclass"):
        dft = df[df["task"] == task]
        if dft.empty:
            continue
        for metric in ("f1_macro", "accuracy", "mcc", "cohen_kappa"):
            piv = dft.pivot_table(index="model", columns="dataset", values=metric)
            piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index) #best models on top
            metric_heatmap(piv, metric, f"{metric} - {task} (all models x datasets)", FIGURES_DIR / f"heatmap_{metric}_{task}.png")
    log.info("Heatmaps generated for %d results", len(results))

'''Per-class F1 figures for multiclass datasets (no LODO)'''    
def per_class_figs(results):
    by_ds = {}
    for r in results:
        if r["task"] != "multiclass" or str(r["dataset"]).startswith("LODO"):
            continue
        cur = by_ds.get(r["dataset"])
        if cur is None or r["f1_macro"] > cur["f1_macro"]: #single pass to pick the best model
            by_ds[r["dataset"]] = r
    for ds, r in by_ds.items():
        class_f1(r["per_class"], ds, r["model"], "multiclass", FIGURES_DIR / f"per_class_{ds}.png")
    log.info("Per-class F1 figures generated for %d datasets: %s", len(by_ds), sorted(by_ds))
    
'''SHAP summary plots for binary classification datasets'''    
def shap_for_dataset(name, results):
    #3 models, TreeExplainer is exact and fast
    cands = [r for r in results if r["dataset"] == name and r["task"] == "binary" and r["model"] in ("RandomForest", "XGBoost", "LightGBM")]
    if not cands:
        return
    best = max(cands, key=lambda r: r["f1_macro"])
    tag = f"{name}__binary__{best['model']}"
    p = MODELS_DIR / f"{tag}.joblib"
    if not p.exists():
        return
    loader, _ = DATASETS[name]
    df = loader()
    X, y, y_bin = build_xy(df)
    del df
    gc.collect()
    _, X_te, _, _, _, _ = split(X, y, y_bin, dataset=name)
    Xs =  X_te.sample(n=min(2000, len(X_te)), random_state=RANDOM_STATE) #SHAP is expensive, sample for speed
    pipe = joblib.load(p)
    pre = pipe.named_steps["pre"] #preprocessor feed the model
    clf = pipe.named_steps["clf"]
    Xt = pre.transform(Xs)
    #prefixed names 
    names = [n.split("__")[-1] for n in pre.get_feature_names_out()]
    inner = getattr(clf, "base", clf) #unwrap LabelEncoderClassifier
    try:
        explainer = shap.TreeExplainer(inner)
        shap_values = explainer.shap_values(Xt, check_additivity=False) #fast and tolerant to missing values
        #variety of outputs by library
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[-1] #binary
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, -1]
        plt.figure()
        shap.summary_plot(shap_values, Xt, feature_names=names, show=False, max_display=20) 
        plt.title(f"SHAP summary for {name} ({best['model']})")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_{name}.png", dpi=300)
        plt.close("all")
        log.info("SHAP summary plot generated for %s (%s)", name, best["model"])
    except Exception as e:
        log.warning("Failed to generate SHAP summary for %s (%s): %s", name, best["model"], e)
    del X, X_te, Xt
    gc.collect()
    
    
'''Produce extended figures for all datasets, including ROC/PR curves and SHAP summaries'''    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-shap", action="store_true") #slowest part
    ap.add_argument("--skip-roc", action="store_true")
    args = ap.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    datasets = available_datasets()
    prog = Progress(3 + len(datasets) * 2, phase='extended_figures') #3 steps + 2 per dataset 
    prog.update(stage='backfill', substage='recomputing metrics from stored matrices')
    results = backfill_metrics()
    prog.advance(stage='heatmaps', substage='metric heatmaps')
    heatmaps(results)
    prog.advance(substage='per-class F1 charts')
    per_class_figs(results)
    prog.advance()
    i_ds = 0
    for name in available_datasets():
        i_ds += 1
        prog.update(stage='curves', dataset=name, substage=f'ROC and PR, dataset {i_ds} of {len(datasets)}')
        if not args.skip_roc:
            try:
                roc_pr_for_datasets(name)
            except Exception as e: #no dataset sink others
                log.warning("Failed to generate ROC/PR curves for %s: %s", name, e)
        prog.advance()
        prog.update(stage='shap', dataset=name, substage=f'feature attribution, dataset {i_ds} of {len(datasets)}')
        if not args.skip_shap:
            try:
                shap_for_dataset(name, results)
            except Exception as e:
                log.warning("Failed to generate SHAP summary for %s: %s", name, e)
        prog.advance()  
    log.info("Extended figures generation completed for %d datasets", len(available_datasets()))
    prog.finish(f'figures complete for {len(datasets)} datasets')
    
if __name__ == "__main__":
    main()
    
