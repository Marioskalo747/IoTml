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
from sklearn.pipeline import Pipeline #preprocessing and classifier (no scaling leakage)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import preprocessing
import models_zoo #patched per seed so the classifiers are reseeded too, not only the split
from config import RESULTS_DIR, SLOW_MODEL_TRAIN_CAP
from evaluation import evaluate_model
from loaders import DATASETS, available_datasets
from models_zoo import model_zoo
from preprocessing import build_xy, make_preprocessor, split, split_meta
from progress import Progress

DEFAULT_SEEDS = [42, 7, 13, 34] #4 seeds mean/std 

#log
log = logging.getLogger("seeds")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(RESULTS_DIR / "seed_study.log", encoding="utf-8")])

'''Stratified trim of training set so <=cap rows'''
def cap_rows(X, y, cap, seed):
    if len(X) <= cap:
        return X, y #already small enough
    idx = (pd.Series(range(len(X)), index=X.index).groupby(y, group_keys=False).apply(lambda g: g.sample(n=max(1, round(len(g) * cap / len(X))), random_state=seed)))
    return X.loc[idx.index], y.loc[idx.index]


PARTIAL = RESULTS_DIR / "seed_study_partial.csv"
MODELS_PER_TASK = 15 #model_zoo returns the same 15 specs for both tasks

'''reload completed experiments so an interrupted run continues instead of starting over'''
def load_partial():
    if not PARTIAL.exists():
        return [], set()
    try:
        d = pd.read_csv(PARTIAL)
    except Exception as e:
        log.warning("could not read %s (%s), starting fresh", PARTIAL.name, e)
        return [], set()
    need = {"dataset", "task", "model", "seed", "f1_macro"}
    if not need <= set(d.columns):
        log.warning("%s has an unexpected format, starting fresh", PARTIAL.name)
        return [], set()
    rows = d.to_dict("records")
    done = {(r["dataset"], r["task"], r["model"], int(r["seed"])) for r in rows}
    return rows, done

'''(dataset, seed, task) groups that already hold a full set of models'''
def complete_groups(rows):
    counts = {}
    for r in rows:
        k = (r["dataset"], int(r["seed"]), r["task"])
        counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n >= MODELS_PER_TASK}

'''datasets whose every seed/task group is already complete (skips the expensive reload)'''
def finished_datasets(rows, seeds, tasks):
    groups = complete_groups(rows)
    out = set()
    for ds in {r["dataset"] for r in rows}:
        if all((ds, s, t) in groups for s in seeds for t in tasks):
            out.add(ds)
    return out

'''Seed study for stable results with different random seeds'''
def main():
    ap = argparse.ArgumentParser() #CLI arguments
    ap.add_argument("datasets", nargs="*") #empty means all available datasets
    ap.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS) #list of random seeds to use
    ap.add_argument("--models", nargs="*", default=None) #list of models to run (default all)
    ap.add_argument("--tasks", nargs="*", default=["multiclass", "binary"]) #list of tasks to evaluate
    args = ap.parse_args()
    wanted = [d for d in (args.datasets or available_datasets()) if d in DATASETS]
    #the study must vary BOTH sources of randomness
    original_seed = preprocessing.RANDOM_STATE
    original_model_seed = models_zoo.RANDOM_STATE
    #resume an interrupted run instead of starting over, the study can be stopped and restarted
    rows, done = load_partial()
    if rows:
        log.info("resume: %d experiments already done, %d (dataset,seed,task) groups complete", len(rows), len(complete_groups(rows)))
    prog = Progress(max(len(wanted) * len(args.seeds), 1), phase="seed_study")

    try:
        skip_ds = finished_datasets(rows, args.seeds, args.tasks)
        for name in wanted:
            #a fully finished dataset is skipped before the loader runs: reloading iot23 alone costs ~3 min
            if name in skip_ds:
                log.info("%s: already complete, skipping (resume)", name)
                for _ in args.seeds:
                    prog.advance()
                continue
            loader, _ = DATASETS[name]
            X, y, y_bin = build_xy(loader()) #load + build features and labels
            gc.collect()
            log.info("%s: %d rows, %d features", name, len(X), X.shape[1])
            #change random seed for each dataset inside preprocessing to split()
            for seed in args.seeds:
                preprocessing.RANDOM_STATE = seed
                #patching random state here reseeds every classifier that accepts a random_state
                models_zoo.RANDOM_STATE = seed
                prog.update(stage="training", dataset=name, model=f"seed={seed}", message=f"{name} seed={seed}")
                X_tr, X_te, y_tr, y_te, yb_tr, yb_te = split(X, y, y_bin, dataset=name)
                #multiclass and binary tasks
                for task in args.tasks:
                    a_tr, a_te = ((y_tr, y_te) if task == "multiclass" else (yb_tr.map({0: "Benign", 1: "Attack"}), yb_te.map({0: "Benign", 1: "Attack"})))
                    labels = sorted(pd.unique(pd.concat([a_tr, a_te]).astype(str))) #union of train/test labels for this task
                    #whole task already finished in an earlier run -> nothing to retrain
                    if (name, seed, task) in complete_groups(rows):
                        log.info("  %s/%s seed=%s: already complete, skipping (resume)", name, task, seed)
                        continue
                    #--models filter
                    for spec in model_zoo(len(labels)):
                        if args.models and spec.name not in args.models:
                            continue
                        #single experiment already done in an earlier run
                        if (name, task, spec.name, seed) in done:
                            continue
                        try:
                            #slow models train on trimmed sample
                            Xt, yt = (cap_rows(X_tr, a_tr, SLOW_MODEL_TRAIN_CAP, seed) if spec.slow else (X_tr, a_tr))
                            #preprocessor is fitted on train only
                            pipe = Pipeline([("pre", make_preprocessor(Xt)), ("clf", spec.build())])
                            #time per seed
                            t0 = time.perf_counter()
                            pipe.fit(Xt, yt.astype(str))
                            dt = time.perf_counter() - t0
                            proba = None
                            try:
                                proba = pipe.predict_proba(X_te) #ROC AUC requires probabilities for some models
                            except Exception:
                                pass
                            r = evaluate_model(pipe, X_te, a_te.astype(str), labels, proba=proba)
                            rows.append({"dataset": name, "task": task, "model": spec.name, "seed": seed, "train_rows": len(Xt), "test_rows": len(X_te), "train_time_s": dt,
                                         #the seed study reports 30 BoT-IoT rows whose test split holds a single class
                                         **split_meta(name, a_tr, a_te), **{k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "mcc", "g_mean")}})
                            done.add((name, task, spec.name, seed))
                            del pipe
                        except Exception as e:
                            log.warning("  FAIL %s/%s/%s seed=%s: %s", name, task, spec.name, seed, e) #fail of a model does not stop the study
                        #checkpoint after every model
                        try:
                            pd.DataFrame(rows).to_csv(PARTIAL, index=False)
                        except OSError:
                            pass #a locked file must never stop the study
                        gc.collect()
                    log.info("seed=%s %s: %d results so far", seed, task, len(rows))
                prog.advance()
            #free memory after each dataset
            del X, y, y_bin
            gc.collect()
    finally:
        preprocessing.RANDOM_STATE = original_seed #restore original random state
        models_zoo.RANDOM_STATE = original_model_seed
    if not rows:
        log.warning("no results")
        return
    #aggregate results and save per dataset, task, model, seed
    d = pd.DataFrame(rows)
    (RESULTS_DIR / "seed_study.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    agg = (d.groupby(["dataset", "task", "model"]).agg(n_seeds=("seed", "nunique"),f1_macro_mean=("f1_macro", "mean"), f1_macro_std=("f1_macro", "std"),
                f1_macro_min=("f1_macro", "min"), f1_macro_max=("f1_macro", "max"),accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                mcc_mean=("mcc", "mean"), mcc_std=("mcc", "std")).reset_index())
    agg.to_csv(RESULTS_DIR / "table_seed_study.csv", index=False)
    PARTIAL.unlink(missing_ok=True) #resume file no longer needed once the phase completed
    log.info("\nVariability of macro-F1 across seeds (maximum standard deviation per dataset):\n%s", agg.groupby("dataset").f1_macro_std.max().round(4).to_string())
    prog.finish(f"seed_study: {len(d)} experiments")
    log.info("Completed: %d experiments, %d seeds", len(d), d.seed.nunique())


if __name__ == "__main__":
    main()
