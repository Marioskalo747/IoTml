#Libraries
import os
import sys
from pathlib import Path

#split() writes its split audit and tuning.py its log to RESULTS_DIR
OUT_NAME = "results_tuningseeds"
os.environ["IOT_IDS_RESULTS_DIR"] = OUT_NAME
SRC = Path(__file__).resolve().parent
(SRC / OUT_NAME).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))

import argparse
import gc
import json
import logging
import time
import warnings

import numpy as np
import pandas as pd
import optuna
from scipy import stats
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import models_zoo
import preprocessing
import tuning
from config import COMMON_FEATURES, PROJECT_DIR, RESULTS_DIR
from evaluation import evaluate_model
from loaders import available_datasets
from preprocessing import make_preprocessor, split
from tuning import SEARCH_SPACE, default_model

MAIN = PROJECT_DIR / "results"          #main run (read only)
EXTRA = PROJECT_DIR / "results_extra"   #extra_studies.py (read only)
OUT = RESULTS_DIR                      
ROWS = OUT / "tuning_seeds.csv"
DEFAULT_SEEDS = [42, 7, 13, 34]         #seeds of seedstudy.py
ANCHOR_SEED = 42
TOL = 1e-6                              #anchor reproduction tolerance

log = logging.getLogger("tuning_seeds")
log.setLevel(logging.INFO)
log.propagate = False
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(OUT / "tuning_seeds.log", encoding="utf-8")):
    h.setFormatter(fmt)
    log.addHandler(h)


'''The pooled 11-feature table '''
def load_combined():
    cache = MAIN / "common_cache"
    frames = [pd.read_pickle(cache / f"{n}.pkl") for n in available_datasets() if (cache / f"{n}.pkl").exists()]
    if len(frames) < 2:
        raise FileNotFoundError(f"{cache} holds fewer than 2 datasets; run runall.py first")
    df = pd.concat(frames, ignore_index=True)
    X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    return X, df["label"].astype(str), df["binary"].astype(int)


'''Tuned hyperparameters and seed-42 scores recorded by extra_studies.py'''
def load_tuned():
    p = EXTRA / "combined_tuning.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run extra_studies.py first")
    out = {}
    for r in json.loads(p.read_text(encoding="utf-8")):
        if r.get("tuned_fallback_to_default"):
            continue  #rejected search
        #fixed best params + seed 42 scores 
        out[(r["task"], r["model"])] = {"params": r["best_params"], "f1_default": r["default"]["f1_macro"], "f1_tuned": r["tuned"]["f1_macro"]}
    return out


'''Every source of randomness follows the seed (split, training cap, default and tuned models)'''
def set_seed(seed):
    preprocessing.RANDOM_STATE = seed
    models_zoo.RANDOM_STATE = seed
    tuning.RANDOM_STATE = seed


def fit_score(clf, X_tr, y_tr, X_te, y_te, labels):
    #same pipeline shape as main
    pipe = Pipeline([("preprocessor", make_preprocessor(X_tr)), ("clf", clf)])
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr.astype(str))
    dt = time.perf_counter() - t0
    res = evaluate_model(pipe, X_te, y_te.astype(str), labels)
    errors = int(len(y_te) - round(res["accuracy"] * len(y_te)))
    return res, dt, errors


def load_rows():
    #previous rows (resume)
    if not ROWS.exists():
        return []
    return pd.read_csv(ROWS).to_dict("records")


def run(args):
    tuned = load_tuned()
    rows = load_rows()
    done = {(r["task"], r["model"], int(r["seed"])) for r in rows}
    #combinations aren't done and have tuned
    todo = [(s, t, m) for s in args.seeds for t in args.tasks for m in args.models if (t, m) in tuned and (t, m, s) not in done]
    if not todo:
        log.info("nothing to run: every (seed, task, model) is already in %s", ROWS.name)
        return rows
    X, y, y_bin = load_combined()
    log.info("combined: %d pooled flows, %d shared features; %d (seed, task, model) to run", len(X), X.shape[1], len(todo))
    for seed in args.seeds:
        jobs = [(t, m) for (s, t, m) in todo if s == seed]
        if not jobs:
            continue
        set_seed(seed)
        #one split per seed 
        X_tr, X_te, y_tr, y_te, yb_tr, yb_te = split(X, y, y_bin, dataset="combined")
        log.info("seed %d: train %d / test %d", seed, len(X_tr), len(X_te))
        for task, model in jobs:
            #multiclass labels or binary mapped to strings
            a_tr, a_te = ((y_tr, y_te) if task == "multiclass" else (yb_tr.map({0: "Benign", 1: "Attack"}), yb_te.map({0: "Benign", 1: "Attack"})))
            labels = sorted(pd.unique(pd.concat([a_tr, a_te]).astype(str)))
            n_classes = len(labels)
            #same constructors as tuning.tune_one
            r_d, t_d, e_d = fit_score(default_model(model, n_classes), X_tr, a_tr, X_te, a_te, labels)
            clf_t = SEARCH_SPACE[model](optuna.trial.FixedTrial(tuned[(task, model)]["params"]), n_classes)
            r_t, t_t, e_t = fit_score(clf_t, X_tr, a_tr, X_te, a_te, labels)
            row = {"seed": seed, "task": task, "model": model, "train_rows": len(X_tr), "test_rows": len(X_te),
                   "f1_default": r_d["f1_macro"], "f1_tuned": r_t["f1_macro"], "delta_f1_macro": r_t["f1_macro"] - r_d["f1_macro"],
                   "mcc_default": r_d["mcc"], "mcc_tuned": r_t["mcc"], "errors_default": e_d, "errors_tuned": e_t, "train_s_default": t_d, "train_s_tuned": t_t}
            rows.append(row)
            pd.DataFrame(rows).to_csv(ROWS, index=False)  #checkpoint after every pair
            log.info("  seed %d %s/%s: default %.4f  tuned %.4f  (%+.4f)  [%.0f s + %.0f s]", seed, task, model, r_d["f1_macro"], r_t["f1_macro"], row["delta_f1_macro"], t_d, t_t)
            gc.collect()
        del X_tr, X_te, y_tr, y_te, yb_tr, yb_te
        gc.collect()
    return rows


'''Seed 42 must give numbers of results_extra/combined_tuning.json'''
def anchor_check(d, tuned):
    #proves rerun match the original
    out = []
    for _, r in d[d.seed == ANCHOR_SEED].iterrows():
        ref = tuned.get((r.task, r.model))
        if ref is None:
            continue
        dd, dt = r.f1_default - ref["f1_default"], r.f1_tuned - ref["f1_tuned"]
        ok = abs(dd) < TOL and abs(dt) < TOL
        out.append({"task": r.task, "model": r.model, "recorded_default": ref["f1_default"], "rerun_default": r.f1_default, "recorded_tuned": ref["f1_tuned"], "rerun_tuned": r.f1_tuned, "reproduced": ok})
        log.info("anchor seed %d %s/%s: default %+.2e, tuned %+.2e -> %s", ANCHOR_SEED, r.task, r.model, dd, dt, "reproduced" if ok else "NOT reproduced")
    return pd.DataFrame(out)


'''mean and spread of default (per task/model)'''
def summarise(d):
    out = []
    for (task, model), g in d.groupby(["task", "model"], sort=False):
        g = g.sort_values("seed")
        delta = g.delta_f1_macro.to_numpy()
        n = len(g)
        sd = float(np.std(delta, ddof=1)) if n > 1 else np.nan
        #paired t-test
        p = float(stats.ttest_rel(g.f1_tuned, g.f1_default).pvalue) if n > 2 and sd > 0 else np.nan
        #half width of 95% confidence interval of mean difference 
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
        #seed to seed spread of default model
        seed_sd = float(np.std(g.f1_default, ddof=1)) if n > 1 else np.nan
        mean = float(delta.mean())
        #gain counts only if every seed agrees and 95% interval of the mean excludes zero
        if n < 3:
            verdict = "too few seeds"
        elif (delta > 0).all() and mean - half > 0:
            verdict = "confirmed gain"
        elif (delta < 0).all() and mean + half < 0:
            verdict = "confirmed loss"
        else:
            verdict = "within noise"
        out.append({"task": task, "model": model, "seeds": n,
                    "f1_default_mean": float(g.f1_default.mean()), "f1_default_std": seed_sd,
                    "f1_tuned_mean": float(g.f1_tuned.mean()), "f1_tuned_std": float(np.std(g.f1_tuned, ddof=1)) if n > 1 else np.nan,
                    "delta_mean": mean, "delta_std": sd, "delta_ci95_low": mean - half, "delta_ci95_high": mean + half,
                    "delta_min": float(delta.min()), "delta_max": float(delta.max()),
                    "seeds_tuned_better": int((delta > 0).sum()), "paired_t_p": p,
                    "delta_seed42": float(g.loc[g.seed == ANCHOR_SEED, "delta_f1_macro"].iloc[0]) if (g.seed == ANCHOR_SEED).any() else np.nan, "verdict": verdict})
    return pd.DataFrame(out)


def plot(d, path):
    #panel per task/model
    groups = list(d.groupby(["task", "model"], sort=False))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.8), squeeze=False)
    for ax, ((task, model), g) in zip(axes[0], groups):
        g = g.sort_values("seed")
        for _, r in g.iterrows():
            c = "#2b8a3e" if r.delta_f1_macro > 0 else "#c92a2a"
            ax.plot([0, 1], [r.f1_default, r.f1_tuned], "-o", color=c, alpha=0.8, lw=1.5, ms=4)
            ax.annotate(str(int(r.seed)), (1.03, r.f1_tuned), fontsize=7, va="center")
        ax.set_xticks([0, 1], ["default", "tuned"])
        ax.set_xlim(-0.2, 1.3)
        ax.set_title(f"{model}, {task}\nΔ = {g.delta_f1_macro.mean():+.4f} ± {g.delta_f1_macro.std(ddof=1):.4f}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("macro-F1 (test)")
    fig.suptitle("Combined dataset: default vs tuned per seed (tuned hyperparameters fixed from seed 42)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--models", nargs="*", default=["RandomForest", "XGBoost"])
    ap.add_argument("--tasks", nargs="*", default=["multiclass", "binary"])
    args = ap.parse_args()
    if ANCHOR_SEED not in args.seeds:
        args.seeds = [ANCHOR_SEED] + args.seeds  #the anchor proves the rerun matches the recorded study
    t0 = time.perf_counter()
    rows = run(args)
    if not rows:
        log.info("no rows")
        return
    d = pd.DataFrame(rows)
    #keep what this invocation asked for (csv may contain rows of earlier run-experiments)
    d = d[d.seed.isin(args.seeds) & d.task.isin(args.tasks) & d.model.isin(args.models)]
    anchor = anchor_check(d, load_tuned())
    anchor.to_csv(OUT / "anchor_seed42.csv", index=False)
    s = summarise(d)
    s.to_csv(OUT / "table_tuning_seeds_summary.csv", index=False)
    plot(d, OUT / "tuning_seeds.png")
    pd.set_option("display.width", 200)
    print("\nsummary (macro-F1 on the test split, tuned hyperparameters fixed from seed 42):")
    print(s[["task", "model", "seeds", "f1_default_mean", "f1_tuned_mean", "delta_mean", "delta_std", "delta_ci95_low", "delta_ci95_high", "seeds_tuned_better", "paired_t_p", "verdict"]].to_string(index=False))
    if len(anchor):
        print(f"\nanchor seed {ANCHOR_SEED}: {int(anchor.reproduced.sum())}/{len(anchor)} reproduced the recorded numbers")
    print(f"\nwritten to {OUT} ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
