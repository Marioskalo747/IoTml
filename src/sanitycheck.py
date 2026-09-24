#Libraries
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ARCHIVES_DIR, MODELS_DIR, RESULTS_DIR

STALL_MINUTES = 70 #minutes without progress.json update considered a stall
#metrics [0,1] range except mcc and kappa [-1,1]
METRICS_01 = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted","precision_macro", "recall_macro", "g_mean", "roc_auc_ovr")

#problems, warnings and notes collected during the checks
problems, warningsm, notes = [], [], []

def P(msg):
    problems.append(msg)
def W(msg):
    warningsm.append(msg)
def N(msg):
    notes.append(msg)

'''json loader with error handling'''
def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None #phase not run yet
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e: #json corrupted
        W(f"{name}:  ({e}) its contents may be corrupted or incomplete")
        return None

'''pipeline state checker'''
def check_progress():
    meta = load("run_meta.json")
    if not meta:
        N("no run_meta.json yet")
        return
    ph = meta.get("phases", {})
    for k, v in ph.items():
        if v.get("status") == "failed":
            P(f"phase '{k}' Failed")
    done = [k for k, v in ph.items() if v.get("status") == "done"]
    run = [k for k, v in ph.items() if v.get("status") == "running"]
    N(f"phases: {len(done)}/{len(ph)} completed, running {run or '—'}")
    #age of progress.json (if it exists) to detect stalls
    pp = RESULTS_DIR / "progress.json"
    if pp.exists() and run:
        age = (time.time() - pp.stat().st_mtime) / 60
        if age > STALL_MINUTES:
            P(f"the progress.json has not been updated in the last {age:.0f} minutes — possible stall")
        else:
            N(f"progress.json: updated {age:.1f} minutes ago")

'''core check'''
def check_results():
    res = load("all_results.json")
    if not res:
        N("no all_results.json yet")
        return
    df = pd.DataFrame([{k: r.get(k) for k in ("dataset", "task", "model", *METRICS_01, "train_rows", "test_rows")} for r in res])
    N(f"all_results.json: {len(df)} records, {df.model.nunique()} models, "
      f"{df.dataset.nunique()} datasets")
    #no metric values outside [0,1]
    for m in METRICS_01:
        if m not in df:
            continue
        v = pd.to_numeric(df[m], errors="coerce")
        bad = df[(v < -1e-9) | (v > 1 + 1e-9)] #tolerance for floating-point noise
        if len(bad):
            P(f"{m}: {len(bad)} values outside [0,1] -> " f"{bad[['dataset','task','model',m]].head(3).to_dict('records')}")
        nan = df[v.isna() & df[m].notna()] #non-numeric values
        if len(nan):
            W(f"{m}: {len(nan)} non-numeric values")
    if "mcc" in df.columns:
        pass #mcc is [-1,1] range, no check for now
    #models that predict only one class (class-imbalanced)
    for r in res:
        cm = r.get("confusion_matrix")
        if not cm:
            continue
        a = np.array(cm)
        if a.ndim == 2 and a.shape[0] > 1 and (a.sum(axis=0) > 0).sum() == 1:
            P(f"{r['dataset']}/{r['task']}/{r['model']}: predicts ONLY one class " f"(class-imbalanced model)")
    #performance worse than (always the most frequent baseline)
    for r in res:
        cm = r.get("confusion_matrix")
        if not cm:
            continue
        a = np.array(cm, dtype=float)
        if a.sum() == 0:
            continue
        majority = a.sum(axis=1).max() / a.sum() #rows = true classes
        if r.get("accuracy") is not None and r["accuracy"] < majority - 1e-3:
            W(f"{r['dataset']}/{r['task']}/{r['model']}: accuracy {r['accuracy']:.4f} < " f"majority class {majority:.4f} — worse than the «always the most frequent»")
    #a tuned model that is worse than its own default
    by_key = {(r["dataset"], r["task"], r["model"]): r for r in res}
    for (ds, task, name), r in by_key.items():
        if not str(name).endswith("__tuned"):
            continue
        base = by_key.get((ds, task, str(name)[: -len("__tuned")]))
        if base is None or r.get("f1_macro") is None or base.get("f1_macro") is None:
            continue
        drop = base["f1_macro"] - r["f1_macro"]
        if drop > 0.05:
            P(f"{ds}/{task}/{name}: macro-F1 {r['f1_macro']:.4f} against {base['f1_macro']:.4f} for the "
              f"untuned {base['model']} (-{drop:.4f}) — the search returned a worse model than the default")

    #missing experiments LODO is excluded from this check
    for (ds, task), g in df[~df.dataset.astype(str).str.startswith("LODO")].groupby(["dataset", "task"]):
        if g.model.nunique() < 10:
            W(f"{ds}/{task}: only {g.model.nunique()} models — expected ~15")
    #results confirmation
    if df.model.astype(str).str.endswith("__tuned").any():
        n = df.model.astype(str).str.endswith("__tuned").sum()
        N(f"{n} tuned models included (D4)")

'''log checker for tracebacks and failures'''
def check_logs():
    hits = []
    for p in sorted(RESULTS_DIR.glob("*.log")):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace") #errors="replace"
        except Exception:
            continue
        n_tb = txt.count("Traceback (most recent call last)")
        n_fail = len(re.findall(r"\bFAIL\b|ΑΠΕΤΥΧΕ|MemoryError", txt))
        if n_tb or n_fail:
            hits.append((p.name, n_tb, n_fail))
    for name, tb, fl in hits:
        #traceback = problems, failure = warnings
        (P if tb else W)(f"{name}: {tb} tracebacks, {fl} failure reports")
    if not hits:
        N("logs: no tracebacks")

'''current run vs previous run comparison'''
def compare_archive(run_id=None):
    if not ARCHIVES_DIR.exists():
        return
    runs = sorted(p for p in ARCHIVES_DIR.glob("run_*") if p.is_dir())
    if not runs:
        return
    src = (ARCHIVES_DIR / run_id) if run_id else runs[-1] #explicit run_id or last run
    old_p = src / RESULTS_DIR.name / "all_results.json"
    new = load("all_results.json")
    if not old_p.exists() or not new:
        return
    try:
        old = json.loads(old_p.read_text(encoding="utf-8"))
    except Exception:
        return
    #identical comparison, key based on (dataset, task, model)
    o = {(r["dataset"], r["task"], r["model"]): r.get("f1_macro") for r in old}
    n = {(r["dataset"], r["task"], r["model"]): r.get("f1_macro") for r in new}
    common = [k for k in n if k in o and o[k] is not None and n[k] is not None]
    if not common:
        return
    d = pd.Series({k: n[k] - o[k] for k in common}) #positive = improvement, negative = regression
    N(f"comparison with {src.name}: {len(common)} common experiments, " f"mean change in f1_macro {d.mean():+.4f}, range [{d.min():+.4f}, {d.max():+.4f}]")
    big = d[d.abs() > 0.15] #large changes from methodology or data (D1/D2) changes
    for k, v in big.items():
        N(f"   large change {k}: {v:+.4f}  (expected after D1/D2)")

'''checks and reports'''
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="only problems")
    ap.add_argument("--compare-archive", default=None)
    args = ap.parse_args()
    check_progress()
    check_results()
    check_logs()
    compare_archive(args.compare_archive)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*72}\nSANITY CHECK  {ts}\n{'='*72}")
    if not args.quiet:
        for m in notes:
            print(f"  ·  {m}")
    for m in warningsm:  #warnings before problems
        print(f"  !!!  {m}")
    for m in problems:
        print(f"  X  {m}")
    if not problems and not warningsm:
        print("  no problems")
    print(f"{'='*72}")
    sys.exit(1 if problems else 0) #problems are considered failure


if __name__ == "__main__":
    main()
