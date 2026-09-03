#Libraries
import argparse
import gc
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")

from config import COMMON_FEATURES, RANDOM_STATE, RESULTS_DIR
from evaluation import evaluate_model
from loaders import DATASETS, available_datasets
from models_zoo import model_zoo
from preprocessing import build_xy, make_preprocessor, split, split_meta
from progress import Progress

#One multiclass model per dataset is the baseline everything here is measured against
SPEC_TRAIN_CAP = 120000  #rows per fit, keeps the phase inside about an hour
GATE_MODELS = ["DecisionTree", "LightGBM"]  #stage 1 candidates: cheap first, strong second
STAGE2_MODEL = "LightGBM"                   #stage 2, and the single-model baseline
SPECIALIST_MODEL = "LightGBM"
MIN_CLASS_TRAIN = 200    #below this a one-vs-rest specialist has nothing to learn
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
TRANSFER_CAP = 40000     #rows per class for the cross-dataset transfer matrix

OUT = RESULTS_DIR / "specialist_study.json"

log = logging.getLogger("specialist")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "specialist_study.log", encoding="utf-8")])


'''Lets the verified metric code in evaluate_model score predictions assembled by hand'''
class _Fixed:
    def __init__(self, preds):
        self.preds = np.asarray(preds, dtype=object)

    def predict(self, X):
        return self.preds


'''Stratified row cap with a per-class floor, so a rare class is not scaled away'''
def cap(X, y, n=SPEC_TRAIN_CAP, floor=MIN_CLASS_TRAIN):
    if len(X) <= n:
        return X, y
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False) .apply(lambda g: g.sample(n=min(len(g), max(round(len(g) * n / len(X)), floor)), random_state=RANDOM_STATE)))
    return X.loc[idx.index], y.loc[idx.index]


'''Fit one model from the zoo inside a pipeline (preprocessor fitted on train only)'''
def fit(name, X, y, n_classes):
    spec = next((s for s in model_zoo(n_classes) if s.name == name), None)
    if spec is None:
        return None
    pipe = Pipeline([("pre", make_preprocessor(X)), ("clf", spec.build())])
    pipe.fit(X, y.astype(str))
    return pipe


'''Probability of the positive class, whatever order the estimator put its classes in'''
def p_attack(pipe, X, positive="Attack"):
    pr = pipe.predict_proba(X)
    classes = list(pipe.classes_)
    return pr[:, classes.index(positive)] if positive in classes else pr[:, -1]



'''whether a cheap binary gate in front of a multiclass model keep the accuracy of one big model'''
def part_a_cascade(X_tr, X_te, y_tr, y_te, yb_tr, yb_te, name, labels, rows):
    attack_tr = y_tr[yb_tr == 1]
    if attack_tr.nunique() < 2:
        log.info("  [A] %s: fewer than 2 attack classes in train, cascade not applicable", name)
        return
    Xa, ya = cap(X_tr.loc[attack_tr.index], attack_tr)
    stage2 = fit(STAGE2_MODEL, Xa, ya, ya.nunique())
    if stage2 is None:
        return
    s2_pred = np.asarray(stage2.predict(X_te), dtype=object)  #computed once, masked per threshold
    del stage2
    gc.collect()

    Xc, yc = cap(X_tr, y_tr)
    single = fit(STAGE2_MODEL, Xc, yc, len(labels))
    if single is not None:
        r = evaluate_model(single, X_te, y_te.astype(str), labels)
        rows.append({"part": "A_cascade", "dataset": name, "gate": "none (single model)", "threshold": None, "sent_to_stage2": 1.0,
                     **{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")}})
        log.info("  [A] %s single model: f1_macro=%.4f", name, r["f1_macro"])
        del single
        gc.collect()

    yb_tr_s = yb_tr.map({0: "Benign", 1: "Attack"})
    for gate_name in GATE_MODELS:
        Xg, yg = cap(X_tr, yb_tr_s)
        try:
            gate = fit(gate_name, Xg, yg, 2)
        except Exception as e:
            log.warning("  [A] %s/%s: gate failed to fit (%s)", name, gate_name, e)
            continue
        if gate is None:
            continue
        thrs = [(t, False) for t in THRESHOLDS]
        try:
            #the threshold that lets 1% of train benign through the test split is never consulted
            ben_idx = yb_tr_s[yb_tr_s == "Benign"].index
            if len(ben_idx):
                q = float(np.quantile(p_attack(gate, X_tr.loc[ben_idx]), 0.99))
                thrs.append((max(q, 1e-9), True))
        except Exception:
            pass
        try:
            pa = p_attack(gate, X_te)
        except Exception:
            log.warning("  [A] %s/%s: gate exposes no usable probabilities", name, gate_name)
            del gate
            gc.collect()
            continue
        te_benign = (np.asarray(yb_te) == 0)
        for thr, calibrated in thrs:
            flags = pa >= thr
            pred = np.full(len(X_te), "Benign", dtype=object)
            if flags.any():
                pred[flags] = s2_pred[flags]
            r = evaluate_model(_Fixed(pred), X_te, y_te.astype(str), labels)
            rows.append({"part": "A_cascade", "dataset": name, "gate": gate_name, "threshold": float(f"{thr:.6g}"), "calibrated_1pct_train_fpr": bool(calibrated),
                         "sent_to_stage2": float(flags.mean()), #what the alert budget actually cost on the test split
                         "test_benign_forwarded": (float(flags[te_benign].mean()) if te_benign.any() else None), **{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc")}})
        log.info("  [A] %s gate=%s: %d operating points measured", name, gate_name, len(thrs))
        del gate
        gc.collect()


'''wether a dedicated one-vs-rest detector beat the shared multiclass boundary for its own class'''
def part_b_specialists(X_tr, X_te, y_tr, y_te, name, labels, rows):
    Xc, yc = cap(X_tr, y_tr)
    single = fit(STAGE2_MODEL, Xc, yc, len(labels))
    if single is None:
        return
    single_pred = np.asarray(single.predict(X_te), dtype=object)
    del single
    gc.collect()
    counts = yc.value_counts()
    y_true = np.asarray(y_te.astype(str))
    for cls in labels:
        n_tr = int(counts.get(cls, 0))
        if n_tr < MIN_CLASS_TRAIN:
            continue
        ovr_tr = pd.Series(np.where(yc.values == cls, cls, "__rest__"), index=yc.index)
        try:
            spec = fit(SPECIALIST_MODEL, Xc, ovr_tr, 2)
        except Exception as e:
            log.warning("  [B] %s/%s: specialist failed to fit (%s)", name, cls, e)
            continue
        if spec is None:
            continue
        spec_pred = np.asarray(spec.predict(X_te), dtype=object)
        truth = y_true == cls
        #both models are reduced to the SAME one-vs-rest question, which is what makes them comparable
        a_ok = (single_pred == cls) == truth
        b_ok = (spec_pred == cls) == truth
        n01 = int(np.sum(~a_ok & b_ok))
        n10 = int(np.sum(a_ok & ~b_ok))
        p_val = None
        try:
            from statsmodels.stats.contingency_tables import mcnemar
            table = [[int(np.sum(a_ok & b_ok)), n10], [n01, int(np.sum(~a_ok & ~b_ok))]]
            exact = (n01 + n10) < 25
            p_val = float(mcnemar(table, exact=exact, correction=not exact).pvalue)
        except Exception:
            pass

        def prf(pred_pos):
            tp = int(np.sum(pred_pos & truth))
            fp = int(np.sum(pred_pos & ~truth))
            fn = int(np.sum(~pred_pos & truth))
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

        p1, r1, f1 = prf(single_pred == cls)
        p2, r2, f2 = prf(spec_pred == cls)
        rows.append({"part": "B_specialist", "dataset": name, "class": cls,
                     "train_rows_class": n_tr, "test_rows_class": int(truth.sum()),
                     "single_precision": p1, "single_recall": r1, "single_f1": f1,
                     "specialist_precision": p2, "specialist_recall": r2, "specialist_f1": f2,
                     "delta_f1": f2 - f1, "mcnemar_p": p_val,
                     "only_single_correct": n10, "only_specialist_correct": n01})
        log.info("  [B] %s/%s: single f1=%.4f specialist f1=%.4f (%+.4f, p=%s)", name, cls, f1, f2, f2 - f1, f"{p_val:.3g}" if p_val is not None else "n/a")
        del spec
        gc.collect()


'''for each attack class, which dataset is the best place to learn it from'''
def part_c_transfer(rows):
    cache = RESULTS_DIR / "common_cache"
    frames = {n: pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()}
    if len(frames) < 2:
        log.warning("  [C] need at least two cached datasets, found %d", len(frames))
        return
    for n, d in frames.items():
        d[COMMON_FEATURES] = (d[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
        frames[n] = d[~d.duplicated(subset=COMMON_FEATURES + ["label"])]  #same dedup rule as split()
    for target, tgt in frames.items():
        for source, src in frames.items():
            if source == target:
                continue
            shared = sorted(set(src["label"]) & set(tgt["label"]))
            if len(shared) < 2:
                log.info("  [C] %s -> %s: fewer than 2 shared classes, skipped", source, target)
                continue
            s = src[src["label"].isin(shared)]
            t = tgt[tgt["label"].isin(shared)]
            s = s.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), TRANSFER_CAP), random_state=RANDOM_STATE))
            clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
            clf.fit(s[COMMON_FEATURES], s["label"])
            r = evaluate_model(clf, t[COMMON_FEATURES], t["label"].astype(str), shared)
            for pc in r["per_class"]:
                if pc["support"] == 0:
                    continue  #the class exists in the source but has no rows in this target
                rows.append({"part": "C_transfer", "source": source, "target": target,"class": pc["class"], "support": pc["support"],
                             "precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"],"n_train": int(len(s))})
            log.info("  [C] %s -> %s: %d shared classes, f1_macro=%.4f", source, target, len(shared), r["f1_macro"])
            del clf
            gc.collect()


def main():
    ap = argparse.ArgumentParser(description="Cascade, one-vs-rest specialists and per-class cross-dataset transfer.")
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--parts", nargs="*", default=["A", "B", "C"])
    args = ap.parse_args()
    wanted = [d for d in (args.datasets or available_datasets()) if d in DATASETS]
    rows: list[dict] = []
    #carry forward the parts this invocation is not recomputing
    kept: list[dict] = []
    if OUT.exists():
        try:
            for r in json.loads(OUT.read_text(encoding="utf-8")):
                part = r.get("part")
                if part in ("A_cascade", "B_specialist") and r.get("dataset") not in wanted:
                    kept.append(r)
                elif part == "C_transfer" and "C" not in args.parts:
                    kept.append(r)
            if kept:
                log.info("keeping %d rows from the previous run", len(kept))
        except Exception as e:
            log.warning("could not reuse %s (%s); it will be rebuilt from this run only", OUT.name, e)
    prog = Progress(max(len(wanted) + 1, 1), phase="specialist_study")

    for name in wanted:
        prog.update(stage="loading", dataset=name, message=f"loading {name}")
        try:
            X, y, yb = build_xy(DATASETS[name][0]())
        except Exception as e:
            log.warning("skip %s: %s", name, e)
            prog.advance()
            continue
        X_tr, X_te, y_tr, y_te, yb_tr, yb_te = split(X, y, yb, dataset=name)
        del X, y, yb
        gc.collect()
        labels = sorted(pd.unique(pd.concat([y_tr, y_te]).astype(str)))
        meta = split_meta(name, y_tr, y_te)
        log.info("%s: train %d / test %d, %d classes, split=%s evaluable=%s", name, len(X_tr), len(X_te), len(labels), meta["split_mode"], meta["evaluable"])
        before = len(rows)
        try:
            if "A" in args.parts:
                prog.update(stage="cascade", dataset=name, message=f"cascade {name}")
                part_a_cascade(X_tr, X_te, y_tr, y_te, yb_tr, yb_te, name, labels, rows)
            if "B" in args.parts:
                prog.update(stage="specialists", dataset=name, message=f"specialists {name}")
                part_b_specialists(X_tr, X_te, y_tr, y_te, name, labels, rows)
        except Exception as e:
            log.exception("FAIL %s: %s", name, e)
        for r in rows[before:]:
            r.update(meta)  #every row carries the split protocol and whether the cell is evaluable
        del X_tr, X_te, y_tr, y_te, yb_tr, yb_te
        gc.collect()
        prog.advance()

    if "C" in args.parts:
        prog.update(stage="transfer", message="per-class cross-dataset transfer")
        try:
            part_c_transfer(rows)
        except Exception as e:
            log.exception("FAIL transfer: %s", e)
    prog.advance()

    rows = kept + rows
    if not rows:
        log.warning("no specialist results")
        prog.finish("specialist_study: nothing to write")
        return
    OUT.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    d = pd.DataFrame(rows)
    for part, fname in (("A_cascade", "table_cascade.csv"),("B_specialist", "table_specialists.csv"),("C_transfer", "table_class_transfer.csv")):
        sub = d[d.part == part].dropna(axis=1, how="all")
        if not sub.empty:
            sub.to_csv(RESULTS_DIR / fname, index=False)
            log.info("written %s (%d rows)", fname, len(sub))
    #headline: for each (target, class), the source that transfers best
    tr = d[d.part == "C_transfer"] if "part" in d.columns else pd.DataFrame()
    if not tr.empty:
        best = tr.loc[tr.groupby(["target", "class"])["f1"].idxmax()]
        best[["target", "class", "source", "f1", "support"]].to_csv(RESULTS_DIR / "table_best_source_per_class.csv", index=False)
        log.info("\nbest source per (target, class):\n%s",best[["target", "class", "source", "f1"]].round(4).to_string(index=False))
    prog.finish(f"specialist_study: {len(rows)} rows")
    log.info("Completed: %d rows", len(rows))


if __name__ == "__main__":
    main()
