#Libraries
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from progress import Progress
from config import RESULTS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid", font_scale=0.95)

#metrics to include in tables and plots
METRICS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro","f1_macro", "f1_weighted", "mcc", "cohen_kappa", "g_mean", "roc_auc_ovr", "train_time_s", "predict_us_per_sample"]

'''reads results by runall'''
def load():
    with open(RESULTS_DIR / "all_results.json", encoding="utf-8") as fh:
        return json.load(fh)
    
'''overview tables of results and returns a dataframe'''
def overview_tables(res: list[dict]):
    #LODO results are excluded from overview tables
    keep = ("dataset", "task", "model", "family", "train_rows", "train_rows_available", "capped",
            "train_cap", "test_rows", "n_features", "split_mode", "evaluable", "n_test_classes",
            "classes_missing_from_train", "n_iter", "converged",
            #IoT-23 puts 123 DDoS rows in train against 23,155 in test
            "min_train_per_class", "classes_undertrained", "fully_learnable", "train_min_class_rows")
    rows = [{**{k: r.get(k) for k in keep}, **{k: r.get(k) for k in METRICS}}
            for r in res if not str(r["dataset"]).startswith("LODO")]
    df = pd.DataFrame(rows)
    df["tuned"] = df["model"].astype(str).str.endswith("__tuned") #flag tuned models vs default 
    df["base_model"] = df["model"].astype(str).str.replace("__tuned", "", regex=False) #name without suffix
    for task in ("multiclass", "binary"):
        d = df[df.task == task].drop(columns=["task"]) #task column is redundant 
        d.to_csv(RESULTS_DIR / f"table_overview_{task}.csv", index=False)
    return df

'''Chmapion model ranked by F1 macro'''
def best_models_table(df: pd.DataFrame):
    #idxmax selects one row and keeps it whole
    champ = df.loc[df.groupby(["dataset", "task"])["f1_macro"].idxmax()].reset_index(drop=True)
    #split_mode/evaluable/fully_learnable with the champion too
    cols = ["dataset", "task", "model", "tuned", "split_mode", "evaluable", "fully_learnable",
            "accuracy", "f1_macro", "f1_weighted", "mcc", "roc_auc_ovr", "train_time_s", "predict_us_per_sample"]
    cols = [c for c in cols if c in champ.columns]
    champ[cols].to_csv(RESULTS_DIR / "table_best_models.csv", index=False) #table with default hyperparameters
    if "tuned" in df.columns:
        _d = df[~df.tuned]
        (_d.loc[_d.groupby(["dataset", "task"])["f1_macro"].idxmax()][cols] .to_csv(RESULTS_DIR / "table_best_models_default_only.csv", index=False))
    return champ

'''Detailed per-class tables for champion models'''
def per_class_tables(res: list[dict], champ: pd.DataFrame):
    for _, row in champ[champ.task == "multiclass"].iterrows(): #champion's full record
        r = next(x for x in res if x["dataset"] == row.dataset and x["task"] == "multiclass" and x["model"] == row.model)
        pc = pd.DataFrame(r["per_class"])
        pc["f1"] = 2 * pc.precision * pc.recall / (pc.precision + pc.recall).replace(0, np.nan) #recompute f1 to avoid rounding issues
        pc.to_csv(RESULTS_DIR / f"table_per_class_{row.dataset}.csv", index=False)
        
'''table of LODO results'''
def lodo_table(res: list[dict]):
    #names of LODO holdout datasets
    rows = [{"holdout": r["dataset"].replace("LODO_holdout_", ""), "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"], "f1_macro": r["f1_macro"],
             "f1_weighted": r["f1_weighted"], "mcc": r["mcc"]} for r in res if str(r["dataset"]).startswith("LODO")]
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / "table_lodo.csv", index=False)

'''accuracy vs training cost'''        
def tradeoff_plot(df: pd.DataFrame):
    #averages per model across datasets, evaluable cells only
    d = df[(df.task == "multiclass") & (df.evaluable != False)]
    d = d.groupby("model", as_index=False).agg(f1=("f1_macro", "mean"), t=("train_time_s", "mean"), pred=("predict_us_per_sample", "mean"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(d.t, d.f1, s=np.sqrt(d.pred.clip(lower=1))*8, alpha=0.7, c=range(len(d)), cmap="tab20")
    for i, row in d.iterrows():
        ax.annotate(row.model, (row.t, row.f1), fontsize=8, xytext=(4,4), textcoords="offset points")
    ax.set_xlabel("Average training time (s)")
    ax.set_ylabel("F1 macro")
    ax.set_xscale("log")
    ax.set_title("Average training time vs F1 macro (multiclass, evaluable datasets only)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "tradeoff_training_time_f1.png", dpi=300)
    plt.close(fig)
    
'''F1 macro heatmap'''
def heatmap_models_datasets(df: pd.DataFrame):
    d = df[(df.task == "multiclass") & (df.evaluable != False)].pivot_table(index="model", columns="dataset", values="f1_macro")
    #rank on the datasets every model actually has
    common = d.dropna(axis=1, how="any")
    order = (common if not common.empty else d).mean(axis=1).sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(d.loc[order], annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "F1-macro"})
    ax.set_title("F1-macro per model and dataset (multiclass, evaluable only)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "heatmap_f1_models_datasets.png", dpi=300)
    plt.close(fig)
    
'''cost table of training time and prediction time per sample'''  
def times_table(df: pd.DataFrame):
    #grouped by task as well as model
    d = df[df.evaluable != False]
    d = (d.groupby(["model", "task"], as_index=False)
           .agg(n_cells=("f1_macro", "size"),
                mean_train_s=("train_time_s", "mean"),
                predict_us_per_sample=("predict_us_per_sample", "mean"),
                mean_f1_macro=("f1_macro", "mean"),
                mean_train_rows=("train_rows", "mean"),
                any_capped=("capped", "max")))
    d.sort_values(["task", "mean_f1_macro"], ascending=[True, False]).to_csv(RESULTS_DIR / "table_model_times.csv", index=False)


'''Flat dump of every record'''
def summary_csv(res: list[dict]):
    flat = pd.DataFrame([{k: v for k, v in r.items() if k not in ("per_class", "confusion_matrix", "labels", "params")} for r in res])
    flat.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return flat

'''tables and figures from all_results.json'''
def main():
    prog = Progress(6, phase='report_assets')
    prog.update(stage='loading', substage='reading all_results.json')
    res = load()
    prog.advance(stage='tables', substage='overview tables')
    df = overview_tables(res)
    prog.advance(substage='champion per dataset')
    champ = best_models_table(df)
    prog.advance(substage='per-class tables')
    per_class_tables(res, champ)
    lodo_table(res)
    prog.advance(stage='figures', substage='model x dataset heatmap')
    heatmap_models_datasets(df)
    prog.advance(substage='accuracy against training cost')
    tradeoff_plot(df)
    times_table(df)
    summary_csv(res)
    prog.finish("report tables and figures written")
    #winning models per dataset and task
    print("report assets written to", RESULTS_DIR)
    print(champ[["dataset", "task", "model", "accuracy", "f1_macro"]].to_string(index=False))
    
if __name__== "__main__":
    main()