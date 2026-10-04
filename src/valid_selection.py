#Libraries
import os
import sys
from pathlib import Path

#preprocessing.split() writes its split audit to RESULTS_DIR
OUT_NAME = "results_validated"
os.environ["IOT_IDS_RESULTS_DIR"] = OUT_NAME
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import gc
import json
import logging
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from config import COMMON_FEATURES, MODELS_DIR, PROJECT_DIR, RANDOM_STATE, RESULTS_DIR
from evaluation import evaluate_model
from loaders import DATASETS, available_datasets
from models_zoo import model_zoo
from preprocessing import build_xy, make_preprocessor, split

MAIN = PROJECT_DIR / "results"            #results of the main run (read only)
TEMPORAL = PROJECT_DIR / "results_temporal"
OUT = RESULTS_DIR                          
OUT.mkdir(parents=True, exist_ok=True)

FPR_TARGETS = (0.05, 0.01, 0.001, 0.0001)  #same as operatingpoints.py
PREVALENCES = (0.5, 0.01, 0.001, 0.0001)
FLOWS_PER_DAY = 10_000_000
FOLDS = 5
CV_MODELS = ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"]
ABLATION_ROWS = 60000  
CV_CAP = 60000         
DS_ORDER = ["bot_iot", "ton_iot", "ciciot2023", "iot23", "combined"]

log = logging.getLogger("validated")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
for h in (logging.StreamHandler(), logging.FileHandler(OUT / "validated_selection.log", encoding="utf-8")):
    h.setFormatter(fmt)
    log.addHandler(h)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, obj):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


'''built exactly as the main run builds them'''
def dataset_xy(name):
    if name == "combined":
        #same pooled table as runall.run_combined 
        cache = MAIN / "common_cache"
        frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
        if not frames:
            raise FileNotFoundError(f"{cache} is empty")
        df = pd.concat(frames, ignore_index=True)
        X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
        return X, df["label"].astype(str), df["binary"].astype(int)
    loader, _ = DATASETS[name]
    return build_xy(loader())


'''Slow models were trained on a 60k subsample'''
def slow_names():
    return {s.name for s in model_zoo(2) if s.slow}


def ppv(tpr, fpr, prevalence):
    denom = tpr * prevalence + fpr * (1.0 - prevalence)
    return float(tpr * prevalence / denom) if denom > 0 else 0.0


'''Threshold with the highest TPR whose FPR <= target'''
def pick_threshold(y01, score, target):
    fpr, tpr, thr = roc_curve(y01, score)
    ok = np.where(fpr <= target)[0] #ROC points for FPR budget
    i = ok[-1] if len(ok) else 0
    return float(thr[i]), float(tpr[i]), float(fpr[i])


def rates(y01, score, threshold):
    #TPR, FPR, absolute false-positive of fixed threshold
    pred = score >= threshold
    pos, neg = y01 == 1, y01 == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    fpr = float(pred[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr, int(pred[neg].sum())

'''Attack probability'''
def attack_score(pipe, X):
    pr = pipe.predict_proba(X)
    return pr[:, list(pipe.classes_).index("Attack")]


#thresholds
def part_thresholds(datasets, models_filter):
    out_path = OUT / "operating_points_validated.json"
    #resume
    rows = load_json(out_path) if out_path.exists() else []
    done = {(r["dataset"], r["model"]) for r in rows}
    all_res = load_json(MAIN / "all_results.json")
    #recorded binary results
    recorded = {(r["dataset"], r["model"]): r for r in all_res if r.get("task") == "binary"}
    oracle = {(r["dataset"], r["model"], r["fpr_target"]): r for r in load_json(MAIN / "operating_points.json")}
    slow = slow_names()

    for name in datasets:
        paths = sorted(MODELS_DIR.glob(f"{name}__binary__*.joblib"))
        todo = []
        for p in paths:
            model = p.stem.split("__", 2)[2] #XGBoost or XGBoost__tuned
            base = model.replace("__tuned", "")
            if base in slow or (models_filter and base not in models_filter and model not in models_filter):
                continue   #exlude slow models 
            if (name, model) in done:
                log.info("skip %s/%s: already in %s", name, model, out_path.name)
                continue
            todo.append((model, p))
        if not todo:
            continue
        log.info("%s: loading and splitting (same split as the main run)", name)
        X, y, y_bin = dataset_xy(name)
        X_tr, X_te, _, _, yb_tr, yb_te = split(X, y, y_bin, dataset=name)
        del X, y, y_bin
        gc.collect()
        #string labels for fit/predict
        ytr_s = yb_tr.map({0: "Benign", 1: "Attack"}).astype(str)
        yte_s = yb_te.map({0: "Benign", 1: "Attack"}).astype(str)
        ytr01, yte01 = yb_tr.to_numpy().astype(int), yb_te.to_numpy().astype(int)
        n_neg_tr, n_neg_te = int((ytr01 == 0).sum()), int((yte01 == 0).sum())
        log.info("%s: train %d (%d benign), test %d (%d benign)", name, len(ytr01), n_neg_tr, len(yte01), n_neg_te)
        #same 5 folds
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(skf.split(X_tr, ytr_s))

        for model, path in todo:
            t0 = time.perf_counter()
            pipe = joblib.load(path)
            #the test split must give the recorded accuracy of the saved model
            acc = float((pipe.predict(X_te) == yte_s.to_numpy()).mean())
            rec_acc = recorded.get((name, model), {}).get("accuracy")
            if rec_acc is None or abs(acc - rec_acc) > 1e-9:
                log.error("%s/%s: test accuracy %.10f does not match the recorded %s -> split differs, skipped", name, model, acc, rec_acc)
                continue
            s_te = attack_score(pipe, X_te) #test scores of saved model
            #out-of-fold scores on the training split
            s_oof = np.empty(len(X_tr))
            for k, (a, b) in enumerate(folds):
                m = clone(pipe)  #same hyperparameters
                m.fit(X_tr.iloc[a], ytr_s.iloc[a])
                s_oof[b] = attack_score(m, X_tr.iloc[b])
                del m
                gc.collect()
            for target in FPR_TARGETS:
                #threshold chosen on train
                thr, tpr_v, fpr_v = pick_threshold(ytr01, s_oof, target)
                tpr_t, fpr_t, fp_t = rates(yte01, s_te, thr)
                #floor at 1/n_negatives
                f_eff = max(fpr_t, 1.0 / n_neg_te) if n_neg_te else fpr_t
                o = oracle.get((name, model, target), {})
                rec = {"dataset": name, "model": model, "fpr_target": target, "threshold_selected_on": f"train_oof_{FOLDS}fold",
                       "threshold": thr, "val_tpr": tpr_v, "val_fpr": fpr_v, "n_negatives_val": n_neg_tr,
                       "selectable": bool(target >= 1.0 / n_neg_tr),
                       "test_tpr": tpr_t, "test_fpr": fpr_t, "test_false_positives": fp_t, "n_negatives_test": n_neg_te,
                       "measurable": bool(target >= 1.0 / n_neg_te), "target_met_on_test": bool(fpr_t <= target),
                       "fpr_used_for_projection": f_eff, "false_alerts_per_day": float(f_eff * FLOWS_PER_DAY),
                       **{f"ppv_at_prev_{p}": ppv(tpr_t, f_eff, p) for p in PREVALENCES},
                       "oracle_tpr": o.get("tpr"), "oracle_fpr": o.get("actual_fpr"), "oracle_threshold": o.get("threshold"),
                       "tpr_drop_vs_oracle": (o["tpr"] - tpr_t) if o.get("tpr") is not None else None,
                       "test_accuracy_check": acc}
                rows.append(rec)
            atomic_write(out_path, rows)  #checkpoint per model
            log.info("%s/%s done in %.0fs: test TPR at 1%% FPR target = %.4f (oracle %.4f)", name, model, time.perf_counter() - t0,
                     next(r["test_tpr"] for r in rows if r["dataset"] == name and r["model"] == model and r["fpr_target"] == 0.01), oracle.get((name, model, 0.01), {}).get("tpr", float("nan")))
            del pipe
            gc.collect()
        del X_tr, X_te
        gc.collect()
    threshold_tables(rows)


def threshold_tables(rows):
    if not rows:
        return
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "table_operating_points_validated.csv", index=False)
    #model per dataset and target on validation 
    best = []
    for (ds, t), g in d.groupby(["dataset", "fpr_target"], sort=False):
        #highest validation TPR -> lowest validation FPR -> name 
        g = g.sort_values(["val_tpr", "val_fpr", "model"], ascending=[False, True, True])
        b = g.iloc[0]
        #model would pick the oracle
        o = g.dropna(subset=["oracle_tpr"])
        ob = o.loc[o.oracle_tpr.idxmax()] if len(o) else None
        best.append({"dataset": ds, "fpr_target": t, "model_chosen_on_val": b.model, "selectable": b.selectable, "measurable": b.measurable,
                     "val_tpr": b.val_tpr, "test_tpr": b.test_tpr, "test_fpr": b.test_fpr, "target_met_on_test": b.target_met_on_test,
                     "ppv_at_prev_0.001": b["ppv_at_prev_0.001"], "false_alerts_per_day": b.false_alerts_per_day,
                     "oracle_model": ob.model if ob is not None else None, "oracle_tpr": ob.oracle_tpr if ob is not None else None,
                     "oracle_fpr": ob.oracle_fpr if ob is not None else None})
    b = pd.DataFrame(best)
    b["_o"] = b.dataset.map({x: i for i, x in enumerate(DS_ORDER)})
    b.sort_values(["_o", "fpr_target"], ascending=[True, False]).drop(columns="_o").to_csv(OUT / "table_operating_points_validated_best.csv", index=False)
    log.info("written table_operating_points_validated.csv and table_operating_points_validated_best.csv")


#cv
'''extendedstudy.subsample'''
def ext_subsample(X, y, cap, seed=RANDOM_STATE):
    if len(X) <= cap:
        return X, y
    np.random.seed(seed)
    #proportional per-class sample with floor of 5 
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False).apply(lambda x: x.sample(n=max(5, round(len(x) * cap / len(X))), random_state=seed)))
    return X.loc[idx.index], y.loc[idx.index]


'''same sampling, folds, pipeline and metric'''
def cv_one(X_train, y_train, dataset, task, models):
    labels = sorted(pd.unique(y_train.astype(str)))
    Xs, ys = ext_subsample(X_train, y_train, CV_CAP)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    zoo = {s.name: s for s in model_zoo(len(labels))}
    out = []
    for name in models:
        if name not in zoo:
            continue
        scores = []
        for tr, te in skf.split(Xs, ys):
            #preprocessor fitted on the fold's training part only 
            pipe = Pipeline([("preprocessor", make_preprocessor(Xs.iloc[tr])), ("clf", zoo[name].build())])
            pipe.fit(Xs.iloc[tr], ys.iloc[tr].astype(str))
            try:
                proba = pipe.predict_proba(Xs.iloc[te])
            except Exception:
                proba = None #no predict_proba
            scores.append(evaluate_model(pipe, Xs.iloc[te], ys.iloc[te].astype(str), labels, proba=proba)["f1_macro"])
            del pipe
            gc.collect()
        out.append({"kind": "cross_validation", "dataset": dataset, "task": task, "model": name, "variant": "5fold",
                    "f1_macro_mean": float(np.mean(scores)), "f1_macro_std": float(np.std(scores)), "folds": scores, "cv_rows": len(Xs)})
        log.info("[cv] %s/%s %s: %.4f +- %.4f", dataset, task, name, np.mean(scores), np.std(scores))
    return out


def part_cv(check_dataset=None):
    out_path = OUT / "cv_combined.json"
    if check_dataset:
        #reproduce one dataset of extendedstudy part d first (same protocol)
        X, y, y_bin = dataset_xy(check_dataset)
        X_tr, _, y_tr, _, yb_tr, _ = split(X, y, y_bin, dataset=check_dataset)
        X_trc, y_trc = ext_subsample(X_tr, y_tr, ABLATION_ROWS)
        yb_trc = yb_tr.loc[y_trc.index].map({0: "Benign", 1: "Attack"})
        mine = cv_one(X_trc, yb_trc, check_dataset, "binary", ["RandomForest"])[0]
        ref = next(r for r in load_json(MAIN / "extended_results.json") if r["kind"] == "cross_validation" and r["dataset"] == check_dataset and r["task"] == "binary" and r["model"] == "RandomForest")
        diff = max(abs(a - b) for a, b in zip(mine["folds"], ref["folds"]))
        log.info("[cv check] %s binary RandomForest: max fold difference vs extended_results = %.2e", check_dataset, diff)
        if diff > 1e-9:
            raise SystemExit("CV protocol does not reproduce extendedstudy part D, combined CV not run")
    if out_path.exists():
        log.info("cv_combined.json exists, combined CV not rerun")
    else:
        X, y, y_bin = dataset_xy("combined")
        X_tr, _, y_tr, _, yb_tr, _ = split(X, y, y_bin, dataset="combined")
        del X, y, y_bin
        X_trc, y_trc = ext_subsample(X_tr, y_tr, ABLATION_ROWS)  #shared cap, as in extendedstudy.main
        yb_trc = yb_tr.loc[y_trc.index].map({0: "Benign", 1: "Attack"})
        rows = cv_one(X_trc, y_trc, "combined", "multiclass", CV_MODELS) + cv_one(X_trc, yb_trc, "combined", "binary", CV_MODELS)
        atomic_write(out_path, rows)
    cv_tables()


def cv_tables():
    cv = [r for r in load_json(MAIN / "extended_results.json") if r["kind"] == "cross_validation"]
    if (OUT / "cv_combined.json").exists():
        cv += load_json(OUT / "cv_combined.json")
    cv = pd.DataFrame(cv)
    res = pd.DataFrame([r for r in load_json(MAIN / "all_results.json") if not str(r["dataset"]).startswith("LODO")])
    res = res[res.evaluable != False]
    rows = []
    for (ds, task), g in cv.groupby(["dataset", "task"]):
        g = g.sort_values("f1_macro_mean", ascending=False)
        sel = g.iloc[0]
        test = res[(res.dataset == ds) & (res.task == task)].set_index("model")
        default = test[~test.index.str.endswith("__tuned")]
        pool = default.loc[[m for m in g.model if m in default.index]]
        tb_all = default.f1_macro.idxmax()
        tb_pool = pool.f1_macro.idxmax()
        sel_test = float(default.loc[sel.model, "f1_macro"])
        rows.append({"dataset": ds, "task": task, "cv_rows": int(sel.cv_rows),
                     "cv_selected_model": sel.model, "cv_f1_macro_mean": sel.f1_macro_mean, "cv_f1_macro_std": sel.f1_macro_std,
                     "cv_selected_test_f1": sel_test, "test_best_in_cv_pool": tb_pool, "test_best_in_cv_pool_f1": float(pool.loc[tb_pool, "f1_macro"]),
                     "test_best_all_15": tb_all, "test_best_all_15_f1": float(default.loc[tb_all, "f1_macro"]),
                     "optimism_of_test_choice": float(default.loc[tb_all, "f1_macro"]) - sel_test,
                     "same_winner": sel.model == tb_all, "cv_ranking": " > ".join(f"{m} {v:.4f}" for m, v in zip(g.model, g.f1_macro_mean))})
    d = pd.DataFrame(rows)
    d["_o"] = d.dataset.map({x: i for i, x in enumerate(DS_ORDER)})
    d.sort_values(["_o", "task"]).drop(columns="_o").to_csv(OUT / "table_best_models_cv.csv", index=False)
    log.info("written table_best_models_cv.csv (%d rows)", len(d))


#tables
def best_table(res_dir, tag):
    res = [r for r in load_json(res_dir / "all_results.json") if not str(r["dataset"]).startswith("LODO")]
    df = pd.DataFrame(res)
    df["tuned"] = df.model.str.endswith("__tuned")
    rows = []
    for (ds, task), g in df.groupby(["dataset", "task"], sort=False):
        ev = g[g.evaluable != False]
        for variant, gg in (("all", ev), ("default_only", ev[~ev.tuned])):
            if gg.empty:
                reason = "; ".join(sorted(set(str(x) for x in g.not_evaluable_reason.dropna()))) or "no evaluable row"
                rows.append({"dataset": ds, "task": task, "variant": variant, "model": None, "evaluable": False, "not_evaluable_reason": reason, "selected_on": "none (not evaluable)"})
                continue
            b = gg.loc[gg.f1_macro.idxmax()]
            ties = sorted(gg.model[(gg.f1_macro - b.f1_macro).abs() < 1e-12])
            rows.append({"dataset": ds, "task": task, "variant": variant, "model": b.model, "tied_with": ", ".join(m for m in ties if m != b.model),
                         "evaluable": True, "not_evaluable_reason": None, "selected_on": "test split (descriptive)", "split_mode": b.get("split_mode"), "accuracy": b.accuracy, "f1_macro": b.f1_macro, "mcc": b.mcc})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"table_best_models_evaluable_{tag}.csv", index=False)
    log.info("written table_best_models_evaluable_%s.csv", tag)

#rebuild 5 CSVs from JSON already exists 
def part_tables():
    best_table(MAIN, "random")
    if (TEMPORAL / "all_results.json").exists():
        best_table(TEMPORAL, "temporal")
    if (MAIN / "extended_results.json").exists():
        cv_tables()
    if (OUT / "operating_points_validated.json").exists():
        threshold_tables(load_json(OUT / "operating_points_validated.json"))


def main():
    ap = argparse.ArgumentParser(description="Validation-chosen thresholds, CV model choice and evaluable-only best-model tables")
    ap.add_argument("parts", nargs="*", default=["thresholds", "cv", "tables"], choices=["thresholds", "cv", "tables"])
    ap.add_argument("--datasets", nargs="*", default=None, help="default: every dataset found + combined")
    ap.add_argument("--models", nargs="*", default=None, help="restrict the threshold part to these models (default: all non-slow)")
    ap.add_argument("--cv-check", default="ton_iot", help="dataset used to verify the CV protocol before the combined CV ('' to skip)")
    args = ap.parse_args()
    #default datasets
    datasets = args.datasets or [d for d in DS_ORDER if d == "combined" or d in available_datasets()]
    log.info("validated_selection: parts=%s datasets=%s -> %s", args.parts, datasets, OUT)
    if "thresholds" in args.parts:
        part_thresholds(datasets, set(args.models) if args.models else None)
    if "cv" in args.parts:
        part_cv(args.cv_check or None)
    if "tables" in args.parts:
        part_tables()
    log.info("finished")


if __name__ == "__main__":
    main()
