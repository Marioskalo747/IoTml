#Libraries
import gc
import json
import logging
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #render charts to files
import seaborn as sns

#Add root and src to sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMON_FEATURES, FIGURES_DIR, MODELS_DIR, RESULTS_DIR #paths and feature list
from loaders import DATASETS, available_datasets
from preprocessing import build_xy, split
from progress import Progress #live progress


log = logging.getLogger("bench") #phase logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BATCH = 50000 #flows measured per model
REPEAT = 3 #timing runs per model
LATENCY_REPEAT = 50 #single-flow calls, median reported



'''prepares batch of flows for benchmarking'''
def sample_flows(n=BATCH):
    cache = RESULTS_DIR / "common_cache" #cached flow data
    frames = [pd.read_pickle(p) for p in sorted(cache.glob("*.pkl"))] #pkl loading
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(min(n, len(df)), random_state=0) #sample flows
    X = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12)) #11 features, cleaned and clipped
    return X

'''batch of flows in a single dataset native feature space (per-dataset models do not use COMMON_FEATURES)'''
def dataset_flows(name, n=BATCH):
    loader, _ = DATASETS[name]
    X, y, y_bin = build_xy(loader())
    _, X_te, _, _, _, _ = split(X, y, y_bin, dataset=name) #same split and seed as training
    del X, y, y_bin
    gc.collect()
    return X_te.sample(min(n, len(X_te)), random_state=0)

'''How heavy is the model, how long to load, and how fast to predict'''
def bench_model(path: Path, X: pd.DataFrame):
    size_mb = path.stat().st_size / 1e6 #file size in MB
    t0 = time.perf_counter()
    pipe = joblib.load(path) #load model pipeline
    load_s = time.perf_counter() - t0 #counter
    pipe.predict(X.head(500)) #warmup
    times = []
    #run multiple times for stable measurement
    for i in range(REPEAT):
        t0 = time.perf_counter()
        pipe.predict(X)
        times.append(time.perf_counter() - t0)
    best = min(times) #keep faster
    one = X.head(1)
    lat = []
    for i in range(LATENCY_REPEAT):
        t0 = time.perf_counter()
        pipe.predict(one)
        lat.append(time.perf_counter() - t0)
    single_us = float(np.median(lat) * 1e6) #median, robust to scheduler noise
    del pipe
    gc.collect()
    #single_call_overhead_us not latency
    return {"size_mb": size_mb, "load_s": load_s, "us_per_flow": best / len(X) * 1e6,"flows_per_s": len(X) / best,
            "single_call_overhead_us": single_us, "batch_size": int(len(X))}
    

def main():
    #benchmark every trained model
    groups = ["combined"] + [d for d in available_datasets() if d in DATASETS]
    models = sorted(MODELS_DIR.glob("*.joblib"))
    prog = Progress(max(len(models), 1), phase="deployment_bench")
    prog.update(stage="loading", substage="sampling a batch of flows")
    i_m = 0
    out = []

    for group in groups:
        gmodels = sorted(MODELS_DIR.glob(f"{group}__*.joblib"))
        if not gmodels:
            continue
        try:
            X = sample_flows() if group == "combined" else dataset_flows(group)
        except Exception as e:
            log.warning("skip group %s: cannot build a flow batch (%s)", group, e)
            continue
        log.info("benchmarking %s: batch %d flows, %d features, %d models", group, len(X), X.shape[1], len(gmodels))
        #loads each model from joblib and measures size, load time, batch throughput and single-flow latency
        for p in gmodels:
            i_m += 1
            prog.update(stage="benchmark", dataset=group,substage=f"model {i_m} of {len(models)}", message=f"measuring {p.stem}") #progress update
            parts = p.stem.split("__")
            if len(parts) < 3:
                continue
            task, model = parts[1], parts[2]
            try:
                r = bench_model(p, X)
                out.append({"dataset": group, "task": task, "model": model, **r})
                log.info("%s/%s/%s: %.1f MB, load %.3fs, %.2f us/flow batch, %.1f flows/s, %.1f us single-flow",
                         group, task, model, r["size_mb"], r["load_s"], r["us_per_flow"], r["flows_per_s"], r["single_call_overhead_us"]) #info
            except Exception as e:
                log.warning("skip %s: %s", p.name, e) #warning if model fails
            prog.advance()
        del X
        gc.collect()


    (RESULTS_DIR / "deployment_bench.json").write_text(json.dumps(out, indent=1)) #save json results
    df = pd.DataFrame(out)
    df.to_csv(RESULTS_DIR / "deployment_bench.csv", index=False) #write csv results
    if df.empty:
        log.warning("no model could be benchmarked, skipping the figure")
        prog.finish("deployment benchmark: no models")
        return
    d = df[(df.task == "multiclass") & (df.dataset == "combined")].sort_values("flows_per_s", ascending=False)

    #Speed bars per model for multiclass models in log scale
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=d, x="model", y="flows_per_s", ax=ax, color="#4878cf") 
    ax.set_yscale("log")
    ax.set_ylabel("Flows per second")
    ax.set_xlabel("")
    ax.set_title("Deployment Benchmark: Multiclass Models (combined, 11 features, batch throughput)")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "deployment_bench_multiclass.png", dpi=300)
    log.info("Deployment benchmark completed. Results saved to %s", len(out))
    prog.finish(f"deployment benchmark complete: {len(out)} models")   
    
if __name__ == "__main__":
    main()
    