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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMON_FEATURES, MODELS_DIR, RANDOM_STATE, RESULTS_DIR
from loaders import DATASETS, available_datasets
from preprocessing import build_xy, split
from progress import Progress

N_BOOT = 500 #bootstrap resample 95% confidence
#No test-set cap
FPR_TARGETS = (0.01, 0.001) #operating points for detection at FPR

#logs
log = logging.getLogger("stats")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(RESULTS_DIR / "validation_stats.log", encoding="utf-8")])

'''training time, test split, trims in stratified way'''
def testsplit_for(name):
    if name == "combined":
        cache = RESULTS_DIR / "common_cache"
        frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
        df = pd.concat(frames, ignore_index=True)
        X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
        y = df["label"].astype(str)
        y_bin = df["binary"].astype(int)
    else:
        loader, _ = DATASETS[name]
        X, y, y_bin = build_xy(loader())
        #same split and seed 
    #train labels are returned too: the dummy baselines must learn their priors from train, not test
    _, X_te, y_tr, y_te, yb_tr, yb_te = split(X, y, y_bin, dataset=name)
    y_te = y_te.astype(str)
    y_tr = y_tr.astype(str)
    yb_te = yb_te.map({0: "Benign", 1: "Attack"})
    yb_tr = yb_tr.map({0: "Benign", 1: "Attack"})
    return X_te, y_te, yb_te, y_tr, yb_tr

'''macro-F1 straight from a confusion matrix, matching f1_score(average="macro", zero_division=0)'''
def macro_f1_cm(C):
    tp = np.diag(C).astype(float)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    pr = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rc = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(tp), where=(pr + rc) > 0)
    return float(f1.mean())

'''95% bootstrap confidence intervals for f1_macro and accuracy, on the FULL test split'''
def bootstrap_ci(y_true, y_pred, labels, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    C = confusion_matrix(yt, yp, labels=labels).astype(float)
    n = int(C.sum())
    if n == 0:
        z = float("nan")
        return {"f1_macro": z, "f1_macro_lo": z, "f1_macro_hi": z, "accuracy": z, "accuracy_lo": z, "accuracy_hi": z, "n_test": 0}
    draws = rng.multinomial(n, (C / n).reshape(-1), size=n_boot).reshape(n_boot, *C.shape)
    f1s = np.array([macro_f1_cm(c) for c in draws])
    accs = np.trace(draws, axis1=1, axis2=2) / n
    #2.5% and 97.5% percentiles for bounds of bootstrap
    #the split the interval refers to, so a reader can never again mistake it for a different one
    return {"f1_macro": macro_f1_cm(C), "f1_macro_lo": float(np.percentile(f1s, 2.5)), "f1_macro_hi": float(np.percentile(f1s, 97.5)),
    "accuracy": float(np.trace(C) / n), "accuracy_lo": float(np.percentile(accs, 2.5)), "accuracy_hi": float(np.percentile(accs, 97.5)),"n_test": n}
    

'''McNemar test for two classifiers on the same dataset'''
def mcnemar_pair(y_true, pred_a, pred_b):
    #CatBoost.predict returns (n, 1) for multiclass while every other estimator returns (n,)
    y_true = np.asarray(y_true).ravel()
    a_ok = np.asarray(pred_a).ravel() == y_true
    b_ok = np.asarray(pred_b).ravel() == y_true
    n01 = int(np.sum(~a_ok & b_ok)) #only b correct
    n10 = int(np.sum(a_ok & ~b_ok)) #only a correct
    table = [[int(np.sum(a_ok & b_ok)), n10], [n01, int(np.sum(~a_ok & ~b_ok))]]
    exact = (n01 + n10) < 25 #fewer discordant pairs, use exact test
    res = mcnemar(table, exact=exact, correction=not exact)
    return {"n_only_a_correct": n10, "n_only_b_correct": n01,"statistic": float(res.statistic) if res.statistic is not None else None,
            "p_value": float(res.pvalue), "test": "exact" if exact else "chi2_corrected"}

'''Detection rate at given FPR operating points'''
def detection_at_fpr(y_true_bin, proba_attack, targets=FPR_TARGETS):
    y = (np.asarray(y_true_bin) == "Attack").astype(int)
    fpr, tpr, thr = roc_curve(y, proba_attack)
    n_neg = int((y == 0).sum())
    #the finest FPR that can be observed at all is 1/n_negatives
    resolution = (1.0 / n_neg) if n_neg else None
    out = {"n_negatives": n_neg, "fpr_resolution": resolution}
    for t in targets:
        ok = np.where(fpr <= t)[0] #every index where FPR is below the target
        i = ok[-1] if len(ok) else 0 #the most sensitive threshold
        out[f"tpr_at_fpr_{t}"] = float(tpr[i])
        #actual threshold
        out[f"threshold_at_fpr_{t}"] = float(thr[i]) if np.isfinite(thr[i]) else None
        out[f"actual_fpr_{t}"] = float(fpr[i])
        #False when the target FPR is finer than one negative sample -> the operating point is not measurable
        out[f"measurable_at_fpr_{t}"] = bool(resolution is not None and t >= resolution)
    return out

'''Trivial baselines for comparison: most frequent, stratified, uniform'''
def baselines(y_tr_like, y_te, labels):
    out = {}
    X_dummy = np.zeros((len(y_te), 1)) #dummy models ignore features
    for strat in ("most_frequent", "stratified", "uniform"):
        d = DummyClassifier(strategy=strat, random_state=RANDOM_STATE)
        d.fit(np.zeros((len(y_tr_like), 1)), y_tr_like)
        p = d.predict(X_dummy)
        out[strat] = {"accuracy": float(accuracy_score(y_te, p)), "f1_macro": float(f1_score(y_te, p, average="macro", labels=labels, zero_division=0))}
    out["majority_class"] = str(pd.Series(y_tr_like).value_counts().idxmax())
    #accuracy of always predicting the train majority class, measured against that class on test
    out["majority_class_accuracy"] = float((pd.Series(y_te).astype(str) == out["majority_class"]).mean())
    #kept separately and named for what it is: the ceiling an oracle knowing the test prior would hit
    out["largest_test_class_share"] = float(pd.Series(y_te).value_counts(normalize=True).max())
    return out


'''Selection bias (cv instead of test) for model selection'''
def selection_bias():
    ext_p = RESULTS_DIR / "extended_results.json"
    all_p = RESULTS_DIR / "all_results.json"
    if not (ext_p.exists() and all_p.exists()):
        log.warning("missing extended_results.json or all_results.json")
        return []
    ext = json.loads(ext_p.read_text(encoding="utf-8"))
    allr = json.loads(all_p.read_text(encoding="utf-8"))
    #extended results cross vaidation
    cv = {}
    for r in ext:
        if r.get("kind") == "cross_validation" and r.get("f1_macro_mean") is not None:
            cv.setdefault((r["dataset"], r["task"]), []).append((r["model"], r["f1_macro_mean"], r.get("f1_macro_std")))
    #main results test set scores
    test = {(r["dataset"], r["task"], r["model"]): r["f1_macro"] for r in allr if not str(r["dataset"]).startswith("LODO")}

    rows = []
    for (ds, task), cands in sorted(cv.items()):
        cv_best, cv_score, cv_std = max(cands, key=lambda t: t[1]) #cv winner
        pool = {m: f for (d, t, m), f in test.items() if d == ds and t == task}
        if not pool or cv_best not in pool:
            continue
        #the honest winner's curse compares like with like
        cv_names = [m for m, _, _ in cands]
        matched = {m: f for m, f in pool.items() if m in cv_names}
        matched_best = max(matched, key=matched.get) if matched else cv_best
        test_best = max(pool, key=pool.get) #best of every trained model (larger pool, not comparable)
        rows.append({"dataset": ds, "task": task,
            "cv_selected_model": cv_best, "cv_f1_macro_mean": cv_score,
            "cv_f1_macro_std": cv_std, "cv_model_test_f1": pool[cv_best],
            #matched-pool figures: this is the number to quote as selection bias
            "matched_best_model": matched_best, "matched_best_f1": matched.get(matched_best),
            "selection_bias_matched": (matched.get(matched_best, pool[cv_best]) - pool[cv_best]),
            "same_winner_matched": cv_best == matched_best,
            "n_candidates_matched": len(matched),
            #unmatched figures kept for reference: best of ALL trained models
            "test_selected_model": test_best, "test_selected_f1": pool[test_best],
            "gap_vs_full_pool": pool[test_best] - pool[cv_best],
            "same_winner": cv_best == test_best, "n_candidates_test": len(pool),
            "n_candidates_cv": len(cands)})
        log.info("  %s/%s: CV->%s (test %.4f) | matched-pool best->%s (%.4f, bias %+.4f) | full-pool best->%s (%.4f, gap %+.4f)",
                 ds, task, cv_best, pool[cv_best], matched_best, rows[-1]["matched_best_f1"],
                 rows[-1]["selection_bias_matched"], test_best, pool[test_best], rows[-1]["gap_vs_full_pool"])
    return rows

'''full-test macro-F1 per model from all_results.json (authoritative ranking, not the capped subsample)'''
def full_test_f1(dataset, task):
    all_p = RESULTS_DIR / "all_results.json"
    if not all_p.exists():
        return {}
    try:
        allr = json.loads(all_p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {r["model"]: r["f1_macro"] for r in allr
            if r.get("dataset") == dataset and r.get("task") == task and r.get("f1_macro") is not None}

'''Statistical tests for each dataset'''
def run_dataset(name, n_boot):
    X_te, y_multi, y_binary, y_multi_tr, y_binary_tr = testsplit_for(name)
    train_labels = {"multiclass": y_multi_tr, "binary": y_binary_tr}
    results = {}
    for task, y_te in (("multiclass", y_multi), ("binary", y_binary)):
        #the label set must be the train-test union exactly as evaluate_model uses it
        labels = sorted(set(pd.unique(y_te)) | set(pd.unique(train_labels[task])))
        preds, probas = {}, {} #every saved model for dataset/task
        for p in sorted(MODELS_DIR.glob(f"{name}__{task}__*.joblib")):
            model = p.stem.split("__", 2)[2] #__tuned suffix intact
            try:
                pipe = joblib.load(p)
                preds[model] = pipe.predict(X_te)
                if task == "binary":
                    try:
                        pr = pipe.predict_proba(X_te)
                        cls = list(getattr(pipe, "classes_", labels))
                        probas[model] = pr[:, cls.index("Attack")] #positive class column
                    except Exception:
                        pass #model without probabilities
                del pipe
                gc.collect()
            except Exception as e:
                log.warning("skip %s/%s/%s: %s", name, task, model, e)
        if not preds:
            continue #no saved models for this dataset/task
        ci = {m: bootstrap_ci(y_te, pr, labels, n_boot) for m, pr in preds.items()}
        #the pair to test must be the two best models on the FULL test split (all_results.json)
        full = full_test_f1(name, task)
        rank_src = "full_test" if any(m in full for m in preds) else "subsample"
        if rank_src == "full_test":
            order = sorted(preds, key=lambda m: -(full.get(m, float("-inf"))))
        else:
            order = sorted(ci, key=lambda m: -ci[m]["f1_macro"])
        top2 = order[:2]
        #McNemar test for the top 2 models
        mc = mcnemar_pair(y_te, preds[top2[0]], preds[top2[1]]) if len(top2) == 2 else None
        if mc:
            mc.update({"model_a": top2[0], "model_b": top2[1], "f1_a": ci[top2[0]]["f1_macro"], "f1_b": ci[top2[1]]["f1_macro"],
                       #f1 on the full test split, which is what the pair was ranked on
                       "f1_a_full_test": full.get(top2[0]), "f1_b_full_test": full.get(top2[1]), "ranking_source": rank_src, "n_test_mcnemar": int(len(y_te))})
        det = {m: detection_at_fpr(y_te, pr) for m, pr in probas.items()} if task == "binary" else {}
        base = baselines(train_labels[task], y_te, labels) #priors learned on train, scored on test

        results[task] = {"n_test": int(len(y_te)), "labels": labels, "confidence_intervals": ci, "mcnemar_top2": mc, "detection_at_fpr": det, "baselines": base}
        log.info("%s/%s: %d models, top2=%s, p=%s", name, task, len(preds), top2, f"{mc['p_value']:.3g}" if mc else "—")

    del X_te
    gc.collect()
    return results

'''True when an entry carries intervals from the full test split rather than the old subsample'''
def _current_method(entry):
    if not isinstance(entry, dict):
        return True  #_selection_bias and other non-dataset keys are method independent
    for task in entry.values():
        if isinstance(task, dict):
            for ci in task.get("confidence_intervals", {}).values():
                return "n_test" in ci
    #guard against mixing two methods 
    return False

'''Tests and statistics for all datasets'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    wanted = args.datasets or (list(available_datasets()) + ["combined"])
    prog = Progress(len(wanted) + 1, phase="validation_stats") #+1 for selection bias step
    #carry forward datasets this invocation is not recomputing
    out = {}
    prev_path = RESULTS_DIR / "validation_stats.json"
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            out = {k: v for k, v in prev.items() if k not in wanted and _current_method(v)}
            stale = [k for k, v in prev.items() if k not in wanted and not _current_method(v)]
            if stale:
                log.warning("discarding %s: computed with the old 10,000-row subsample. Rerun "
                            "validationstats.py for them so every interval uses the full test split", ", ".join(sorted(stale)))
            if out:
                log.info("keeping previous results for: %s", ", ".join(sorted(k for k in out if not k.startswith("_"))))
        except Exception as e:
            log.warning("could not reuse %s (%s); it will be rebuilt from this run only", prev_path.name, e)
    for name in wanted:
        if name != "combined" and name not in DATASETS:
            prog.advance() #step counts when skipped
            continue
        prog.update(stage="bootstrap_ci", dataset=name, message=f"CI/McNemar/FPR for {name}")
        log.info("%s", name)
        try:
            out[name] = run_dataset(name, args.boot)
        except Exception as e:
            log.warning("FAIL %s: %s", name, e)
        prog.advance()

    prog.update(stage="selection_bias", message="Selection Bias (CV vs test)")
    log.info("Selection Bias (CV vs test)")
    sb = selection_bias()
    if sb:
        pd.DataFrame(sb).to_csv(RESULTS_DIR / "table_selection_bias.csv", index=False)
        out["_selection_bias"] = sb

    (RESULTS_DIR / "validation_stats.json").write_text(
        json.dumps(out, indent=1, default=str), encoding="utf-8")
    #nested structure to plain CSV
    ci_rows, mc_rows, det_rows, bl_rows = [], [], [], []
    for ds, tasks in out.items():
        if ds.startswith("_"):   # _selection_bias list no dict 
            continue
        for task, r in tasks.items():
            for m, c in r["confidence_intervals"].items():
                ci_rows.append({"dataset": ds, "task": task, "model": m, **c})
            if r["mcnemar_top2"]:
                mc_rows.append({"dataset": ds, "task": task, **r["mcnemar_top2"]})
            for m, d in r["detection_at_fpr"].items():
                det_rows.append({"dataset": ds, "task": task, "model": m, **d})
            for strat, v in r["baselines"].items():
                if isinstance(v, dict): #maajority class and share are scalars
                    bl_rows.append({"dataset": ds, "task": task, "baseline": strat, **v})
    for rows, fname in ((ci_rows, "table_confidence_intervals.csv"),(mc_rows, "table_mcnemar.csv"),(det_rows, "table_detection_at_fpr.csv"),
                        (bl_rows, "table_baselines.csv")):
        if rows:
            pd.DataFrame(rows).to_csv(RESULTS_DIR / fname, index=False)
            log.info("Written %s (%d lines)", fname, len(rows))

    prog.finish(f"validation_stats: {len(out)} datasets")
    log.info("Completed: %d datasets", len(out))


if __name__ == "__main__":
    main()
