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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg") #png files only, no GUI
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

#shared constants of study 
from config import (COMMON_FEATURES, MAX_TRAIN_ROWS, MIN_ROWS_PER_CLASS_SLOW, PROJECT_DIR,
                    RANDOM_STATE, RESULTS_DIR, SLOW_MODEL_TRAIN_CAP)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import preprocessing
import tuning
from evaluation import evaluate_model
from loaders import available_datasets, load_ciciot2023
from models_zoo import model_zoo
from preprocessing import build_xy, make_preprocessor, split
from tuning import N_TRIALS, SEARCH_SPACE, TUNE_ROWS, tune_one

DEFAULT_OUT = "results_extra"  # default output folder
MERGED = "DoS_DDoS" #merged class
MERGE_MODELS = ["DecisionTree", "RandomForest", "ExtraTrees", "HistGradientBoosting", "XGBoost", "LightGBM", "CatBoost"]


'''output of this run at the folder'''
def redirect_outputs(out_dir: Path):
    for d in (out_dir, out_dir / "figures", out_dir / "models"):
        d.mkdir(parents=True, exist_ok=True)
    #tune_one() saves the tuned pipeline and the Optuna history 
    tuning.MODELS_DIR = out_dir / "models"
    tuning.FIGURES_DIR = out_dir / "figures"
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler): #the results/tuning.log handler from the import
            root.removeHandler(h)
            h.close()
    fh = logging.FileHandler(out_dir / "extra_studies.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    return out_dir


'''Stratified trim with per-class floor (same rule of runall.subsample)'''
def subsample(X, y, cap):
    if len(X) <= cap:
        return X, y #already small enough
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False)
           .apply(lambda s: s.sample(n=min(len(s), max(round(len(s) * cap / len(X)), MIN_ROWS_PER_CLASS_SLOW)), random_state=RANDOM_STATE)))
    return X.loc[idx.index], y.loc[idx.index]


'''The pooled 11-feature table run_combined trains on'''
def load_combined():
    cache = RESULTS_DIR / "common_cache"
    #polled in available_datasets() order (match the run_combined order)
    frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
    if len(frames) < 2:
        raise FileNotFoundError("common_cache holds fewer than 2 datasets; run runall.py first")
    df = pd.concat(frames, ignore_index=True)
    #same cleaning as main run
    X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    #return features, multiclass label, binary label
    return X, df["label"].astype(str), df["binary"].astype(int)


'''Append the tuned rows to results/all_results.json'''
def merge_into_all_results(records):
    allp = RESULTS_DIR / "all_results.json"
    if not allp.exists():
        print("all_results.json not found: nothing merged")
        return 0
    rows = json.loads(allp.read_text(encoding="utf-8"))
    known = {(r["dataset"], r["task"], r["model"]) for r in rows}
    #family comes from the base model
    families = {s.name: s.family for s in model_zoo(3)}
    added = 0
    for r in records:
        name = r["model"] + "__tuned"
        if (r["dataset"], r["task"], name) in known:
            continue #idempotent 
        rows.append({**r["tuned"], "dataset": r["dataset"], "task": r["task"], "model": name,
                     "family": families.get(r["model"], "boosting"), "params": r["best_params"],
                     "roc_auc_ovr": r.get("tuned_roc_auc"), "per_class": r["per_class_tuned"],
                     "confusion_matrix": r["confusion_matrix_tuned"], "labels": r["labels"],
                     "train_time_s": r.get("train_time_s"),
                     **{k: r.get(k) for k in ("train_rows", "test_rows", "train_rows_available", "capped",
                                              "train_cap", "split_mode", "evaluable", "n_test_classes",
                                              "classes_missing_from_train", "min_train_per_class",
                                              "classes_undertrained", "fully_learnable", "n_features",
                                              "train_min_class_rows", "not_evaluable_reason", "labels_scored",
                                              "n_labels_unscored", "tuned_fallback_to_default")}})
        added += 1
    if added:
        #atomic write
        tmp = allp.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        tmp.replace(allp)
    print(f"all_results.json: {added} tuned rows added, {len(rows)} records in total")
    print("  the models of those rows live in this script's own folder, not in models/, so " "validationstats.py and operatingpoints.py will not see them")
    return added



'''Wether tuning change the champion of the pooled model'''
def champion_table(records, out_dir: Path):
    allp = RESULTS_DIR / "all_results.json"
    if not allp.exists():
        print("all_results.json not found: champion comparison skipped")
        return
    rows = json.loads(allp.read_text(encoding="utf-8"))
    out = []
    for task in ("binary", "multiclass"):
        #the zoo as the main run trained it
        base = {r["model"]: r["f1_macro"] for r in rows if r["dataset"] == "combined" and r["task"] == task and not str(r["model"]).endswith("__tuned")}
        if not base:
            continue
        tuned = {r["model"] + "__tuned": r["tuned"]["f1_macro"] for r in records if r["task"] == task}
        champ_d = max(base, key=base.get) #best model withqout tuning
        pool = {**base, **tuned}  #default zoo and tuned variants
        champ_t = max(pool, key=pool.get) #best model after tuning
        out.append({"task": task, "champion_default": champ_d, "f1_default": base[champ_d],
                    "champion_with_tuning": champ_t, "f1_with_tuning": pool[champ_t],
                    "delta_f1_macro": pool[champ_t] - base[champ_d],
                    "changed": champ_t != champ_d, "n_candidates": len(pool)})
    if not out:
        return
    d = pd.DataFrame(out)
    d.to_csv(out_dir / "table_combined_champion.csv", index=False)
    print("\nchampion of the pooled model, default zoo against default + tuned:")
    print(d.to_string(index=False))
    print("written table_combined_champion.csv")


'''Optuna on the pooled data (same search space and protocol of tuning.py)'''
def part_combined_tuning(args):
    out = args.out_dir / "combined_tuning.json"
    records = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["task"], r["model"]) for r in records}
    X, y, y_bin = load_combined()
    print(f"combined: {len(X):,} pooled flows, {X.shape[1]} shared features, {y.nunique()} classes")
    #same split function as main
    X_tr, X_te, y_tr, y_te, yb_tr, yb_te = split(X, y, y_bin, dataset="combined")
    del X, y, y_bin
    gc.collect()
    #2 targets on same split 
    targets = {"multiclass": (y_tr, y_te), "binary": (yb_tr.map({0: "Benign", 1: "Attack"}), yb_te.map({0: "Benign", 1: "Attack"}))}
    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task] #binary first
    print(f"train {len(X_tr):,} / test {len(X_te):,} {args.trials} trials x 3 folds on {args.tune_rows:,} rows, "
          f"then refit on the full training split")
    for task in tasks:
        for model in args.models:
            if model not in SEARCH_SPACE:
                print(f"skip {model}: no search space in tuning.py")
                continue
            if (task, model) in done:
                print(f"skip combined/{task}/{model}: already in {out.name}")
                continue
            t0 = time.perf_counter()
            try:
                #optuna serach on subsample
                rec = tune_one(model, X_tr, X_te, targets[task][0], targets[task][1], "combined", task, args.trials, args.tune_rows)
            except Exception as e:
                print(f"FAIL combined/{task}/{model}: {e}")
                continue
            records.append(rec)
            out.write_text(json.dumps(records, indent=1, default=str), encoding="utf-8") #checkpoint
            d = rec["tuned"]["f1_macro"] - rec["default"]["f1_macro"]
            print(f"combined/{task}/{model}: default={rec['default']['f1_macro']:.4f} "
                  f"tuned={rec['tuned']['f1_macro']:.4f} ({d:+.4f}), cv={rec['best_cv_f1_macro']:.4f}, "
                  f"{(time.perf_counter() - t0) / 60:.0f} min"
                  + ("  [search rejected, default kept]" if rec["tuned_fallback_to_default"] else ""), flush=True)
            gc.collect()
    if not records:
        print("no combined tuning records")
        return
    #summary table
    pd.DataFrame([{"task": r["task"], "model": r["model"],
                   "f1_default": r["default"]["f1_macro"], "f1_tuned": r["tuned"]["f1_macro"],
                   "delta_f1_macro": r["tuned"]["f1_macro"] - r["default"]["f1_macro"],
                   "best_cv_f1_macro": r["best_cv_f1_macro"], "tune_rows": r["tune_rows"],
                   "train_rows": r["train_rows"], "tune_time_s": r["tune_time_s"],
                   "fallback_to_default": bool(r["tuned_fallback_to_default"])} for r in records]
                 ).to_csv(args.out_dir / "table_combined_tuning.csv", index=False)
    print(f"written {out.name} and table_combined_tuning.csv ({len(records)} records), " f"models in {out.parent.name}/models/")
    champion_table(records, args.out_dir)
    if args.merge_all_results:  #opt-in only 
        merge_into_all_results(records)


'''DoS and DDoS relabelled as one class'''
def merge_labels(y):
    s = pd.Series(y)
    #keep original index
    return pd.Series(np.where(s.isin(["DoS", "DDoS"]), MERGED, s), index=s.index)


'''Fit one zoo model as run_all'''
def fit_zoo(name, X, y, n_classes, slow_cap):
    spec = next((s for s in model_zoo(n_classes) if s.name == name), None)
    if spec is None:
        return None, None, None #unknown model name
    Xt, yt = subsample(X, y, slow_cap) if spec.slow else (X, y)
    #no leakage of test set into the preprocessor
    pipe = Pipeline([("pre", make_preprocessor(Xt)), ("clf", spec.build())])
    t0 = time.perf_counter()
    pipe.fit(Xt, yt.astype(str))
    return pipe, time.perf_counter() - t0, len(Xt) #model, training time, training rows


'''One result row, with the per-class F1'''
def row(model, variant, n_classes, y_true, y_pred, labels, train_s, train_rows, dd_errors=None):
    pc = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0) #F1 per class
    return {"model": model, "variant": variant, "n_classes": n_classes,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
            "errors": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
            "dos_ddos_errors": dd_errors, "train_time_s": train_s, "train_rows": train_rows,
            "per_class_f1": {l: float(v) for l, v in zip(labels, pc)}}


'''Macro-F1 and error count per model and variant of merge study'''
def plot_merge(d: pd.DataFrame, path: Path):
    order = [v for v in ("original_10", "eval_merged_9", "retrained_merged_9") if v in set(d["variant"])]
    d = d[d["variant"].isin(order)].copy()
    d["variant"] = pd.Categorical(d["variant"], order, ordered=True)
    models = list(dict.fromkeys(d["model"])) #already canonical: keep the table's order on the axis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(data=d, x="model", y="f1_macro", hue="variant", hue_order=order, order=models, ax=axes[0])
    axes[0].set_title("Macro-F1: 10 classes against DoS+DDoS merged")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("F1-macro")
    sns.barplot(data=d, x="model", y="errors", hue="variant", hue_order=order, order=models, ax=axes[1])
    axes[1].set_title("Misclassified test flows")
    axes[1].set_ylabel("errors")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(title="", fontsize=8)
    fig.suptitle("CICIoT2023: how much of the deficit is the DoS/DDoS distinction", y=1.02)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


'''CICIoT2023 with DoS+DDoS merged'''
def part_ciciot_merge(args):
    out = args.out_dir / "ciciot_merge_study.json"
    kept = []
    if out.exists():
        try:
            kept = [r for r in json.loads(out.read_text(encoding="utf-8")) if r["model"] not in args.models]
            if kept:
                print(f"keeping {len(kept)} rows of {len({r['model'] for r in kept})} models from the previous run")
        except Exception as e:
            print(f"could not reuse {out.name} ({e}); it will be rebuilt from this run only")
    #split() reads the cap from its own module
    old_cap, old_audit = preprocessing.MAX_TRAIN_ROWS, preprocessing.record_split_audit
    preprocessing.MAX_TRAIN_ROWS = args.train_cap
    preprocessing.record_split_audit = lambda *a, **k: None
    try:
        t0 = time.perf_counter()
        X, y, y_bin = build_xy(load_ciciot2023())
        #split on the original labels
        X_tr, X_te, y_tr, y_te, _, _ = split(X, y, y_bin, dataset="ciciot2023")
        X = y = y_bin = None
        gc.collect()
        y_tr, y_te = y_tr.astype(str), y_te.astype(str)
        my_tr, my_te = merge_labels(y_tr), merge_labels(y_te) #merged labels
        labels10, labels9 = sorted(set(y_tr) | set(y_te)), sorted(set(my_tr) | set(my_te))
        print(f"ciciot2023: train {len(X_tr):,} / test {len(X_te):,} (train cap {args.train_cap:,}, "
              f"slow cap {args.slow_cap:,}), loaded in {time.perf_counter() - t0:.0f}s")
        print(f"test DoS {int((y_te == 'DoS').sum()):,} DDoS {int((y_te == 'DDoS').sum()):,}")
        ref = {}
        allp = RESULTS_DIR / "all_results.json" #read only for the reference column
        if allp.exists():
            ref = {r["model"]: r["f1_macro"] for r in json.loads(allp.read_text(encoding="utf-8")) if r["dataset"] == "ciciot2023" and r["task"] == "multiclass"}
        rows = []
        for name in args.models:
            try:
                #original 10-class model
                pipe, dt, n = fit_zoo(name, X_tr, y_tr, len(labels10), args.slow_cap)
                if pipe is None:
                    print(f"skip {name}: not in the zoo")
                    continue
                #CatBoost returns (n, 1) for multiclass 
                pred = pd.Series(np.asarray(pipe.predict(X_te)).ravel(), index=y_te.index).astype(str)
                #the original variant must agree with evaluate_model
                assert abs(evaluate_model(pipe, X_te, y_te, labels10)["f1_macro"] - f1_score(y_te, pred, average="macro", labels=labels10, zero_division=0)) < 1e-9
                pipe = None
                gc.collect()
                #how much of the error mass is the DoS/DDoS pair alone
                dd = int((((y_te == "DoS") & (pred == "DDoS")) | ((y_te == "DDoS") & (pred == "DoS"))).sum())
                rows.append(row(name, "original_10", 10, y_te, pred, labels10, dt, n, dd))
                #same predictions
                rows.append(row(name, "eval_merged_9", 9, my_te, merge_labels(pred), labels9, dt, n, 0))
                #same model retrained on the merged label
                pipe, dt9, n9 = fit_zoo(name, X_tr, my_tr, len(labels9), args.slow_cap)
                pred9 = pd.Series(np.asarray(pipe.predict(X_te)).ravel(), index=y_te.index).astype(str)
                rows.append(row(name, "retrained_merged_9", 9, my_te, pred9, labels9, dt9, n9, 0))
                pipe = None
                o, e, r = rows[-3], rows[-2], rows[-1]
                #log line per model
                print(f"{name:22s} original f1={o['f1_macro']:.4f} err={o['errors']:6d} "
                      f"(DoS<->DDoS {100 * dd / max(o['errors'], 1):.0f}% of them, all_results {ref.get(name, float('nan')):.4f}) | "
                      f"eval-merged f1={e['f1_macro']:.4f} err={e['errors']:6d} | "
                      f"retrained-merged f1={r['f1_macro']:.4f} err={r['errors']:6d}", flush=True)
                out.write_text(json.dumps(kept + rows, indent=1), encoding="utf-8") #checkpoint per model
            except Exception as ex:
                print(f"FAIL {name}: {type(ex).__name__}: {ex}")
            gc.collect()
    finally:
        preprocessing.MAX_TRAIN_ROWS = old_cap
        preprocessing.record_split_audit = old_audit
    rows = kept + rows
    if not rows:
        print("no merge-study rows")
        return
    #main table and the per-class F1 table
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "per_class_f1"} for r in rows])
    d.to_csv(args.out_dir / "table_ciciot_merge.csv", index=False)
    pc = pd.DataFrame([{"model": r["model"], "variant": r["variant"], **r["per_class_f1"]} for r in rows])
    pc.to_csv(args.out_dir / "table_ciciot_merge_per_class.csv", index=False)
    print("\nmean over models:")
    print(d.groupby("variant")[["accuracy", "f1_macro", "errors"]].mean().round(4).to_string())
    print("\nmean per-class F1 over models:")
    #unaffected classes
    cols = [c for c in labels10 if c not in ("DoS", "DDoS")] + [c for c in (MERGED, "DoS", "DDoS") if c in pc]
    print(pc.groupby("variant")[cols].mean().round(3).T.to_string())
    try:
        plot_merge(d, args.out_dir / "figures" / "ciciot_merge.png")
        print("\nwritten figures/ciciot_merge.png")
    except Exception as e:
        print(f"\nfigure skipped ({e})")  #plot fail
    print(f"written {out.name}, table_ciciot_merge.csv and table_ciciot_merge_per_class.csv")


def main():
    ap = argparse.ArgumentParser(description="Standalone extra studies: pooled-model tuning and the CICIoT2023 DoS/DDoS merge.")
    ap.add_argument("part", nargs="?", default="both", choices=["both", "combined-tuning", "ciciot-merge"])
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"output folder, relative to src/ (default: {DEFAULT_OUT})")
    #Combined tuning
    ap.add_argument("--models", nargs="*", default=None, help="default: the 4 tuned models / the 7 merge-study models")
    ap.add_argument("--task", choices=["multiclass", "binary", "both"], default="both")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--tune-rows", type=int, default=TUNE_ROWS, dest="tune_rows")
    ap.add_argument("--merge-all-results", action="store_true", dest="merge_all_results",
                    help="also append the tuned rows to results/all_results.json (off by default)")
    #CICIoT2023 merge study
    ap.add_argument("--train-cap", type=int, default=MAX_TRAIN_ROWS, dest="train_cap")
    ap.add_argument("--slow-cap", type=int, default=SLOW_MODEL_TRAIN_CAP, dest="slow_cap")
    args = ap.parse_args()
    out_dir = Path(args.out)
    args.out_dir = redirect_outputs(out_dir if out_dir.is_absolute() else PROJECT_DIR / out_dir)
    print(f"every output of this run goes to {args.out_dir}")
    chosen = args.models
    if args.part in ("both", "combined-tuning"):
        args.models = chosen or list(SEARCH_SPACE)
        print(f"A combined tuning: {args.models}")
        part_combined_tuning(args)
    if args.part in ("both", "ciciot-merge"):
        #default for B 
        args.models = chosen or MERGE_MODELS
        print(f"\nB CICIoT2023 DoS/DDoS merge: {args.models}")
        part_ciciot_merge(args)


if __name__ == "__main__":
    main()
