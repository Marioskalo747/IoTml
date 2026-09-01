#Libraries
import argparse 
import json
import os
import subprocess #each phase is run as a separate process
import sys
import time
from datetime import datetime
from pathlib import Path

if "--smoke" in sys.argv:
    os.environ.setdefault("IOT_IDS_RESULTS_DIR", "results_smoke")
    os.environ.setdefault("IOT_IDS_MODELS_DIR", "models_smoke")
    os.environ.setdefault("IOT_IDS_ARCHIVES_DIR", "archives_smoke")

from config import RESULTS_DIR
from archiverun import archive, clear_active
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_data import SMOKE_DATA_DIR, smoke_datasets

ROOT = Path(__file__).resolve().parents[1]
META_PATH = RESULTS_DIR / "run_meta.json"   

#full execution order (sys.executable is the current python interpreter)
PHASES = [
    ("run_all", [sys.executable, str(ROOT / "src" / "runall.py")]),
    ("tuning", [sys.executable, str(ROOT / "src" / "tuning.py")]),
    ("extended_study", [sys.executable, str(ROOT / "src" / "extendedstudy.py")]),
    ("deep_study", [sys.executable, str(ROOT / "src" / "deepstudy.py")]),
    ("anomaly_study", [sys.executable, str(ROOT / "src" / "anomalystudy.py")]),
    ("adaptation_study", [sys.executable, str(ROOT / "src" / "adaptionstudy.py")]),
    ("deployment_bench", [sys.executable, str(ROOT / "src" / "deploymentbench.py")]),
    ("extended_figures", [sys.executable, str(ROOT / "src" / "make_extended_figures.py")]),
    ("report_assets", [sys.executable, str(ROOT / "src" / "reportassets.py")]),
    ("validation_stats", [sys.executable, str(ROOT / "src" / "validationstats.py")]),
    ("domain_shift_study", [sys.executable, str(ROOT / "src" / "domainshift_study.py")]),
    ("anomaly_sweep", [sys.executable, str(ROOT / "src" / "anomalysweep.py")]),
    ("leakage_ablation", [sys.executable, str(ROOT / "src" / "leakage_ablation.py")]),
    ("operating_points", [sys.executable, str(ROOT / "src" / "operatingpoints.py")]),
    #wether weighting the rare attack classes actually help
    ("cost_sensitive", [sys.executable, str(ROOT / "src" / "costsensitive.py")]),
    ("seed_study", [sys.executable, str(ROOT / "src" / "seedstudy.py")]),
    #placed late because it is the most expensive of the optional studies
    ("specialist_study", [sys.executable, str(ROOT / "src" / "specialiststudy.py")]),
    #runs last and reads everything
    ("sanity_check", [sys.executable, str(ROOT / "src" / "sanitycheck.py")])]

#Second protocol, run inside the same pipeline but write its own directories so it does not overwrite the main run
TEMPORAL_ARM_DATASETS = ["bot_iot", "iot23"]
#derived from whatever the base directories are
TEMPORAL_ENV = {"IOT_IDS_SPLIT": "temporal",
                "IOT_IDS_RESULTS_DIR": os.environ.get("IOT_IDS_RESULTS_DIR", "results") + "_temporal",
                "IOT_IDS_MODELS_DIR": os.environ.get("IOT_IDS_MODELS_DIR", "models") + "_temporal"}
TEMPORAL_PHASES = [("temporal_run_all", [sys.executable, str(ROOT / "src" / "runall.py")] + TEMPORAL_ARM_DATASETS + ["--no-pooled"]),
    ("temporal_report_assets", [sys.executable, str(ROOT / "src" / "reportassets.py")]),
    ("temporal_validation_stats", [sys.executable, str(ROOT / "src" / "validationstats.py")] + TEMPORAL_ARM_DATASETS),
    ("temporal_seed_study", [sys.executable, str(ROOT / "src" / "seedstudy.py")] + TEMPORAL_ARM_DATASETS),
    ("temporal_sanity_check", [sys.executable, str(ROOT / "src" / "sanitycheck.py")])]

#only this phase is critical, if it fails the pipeline stops
CRITICAL_PHASES = {"run_all"}
#advisory phases report findings through their exit code
ADVISORY_PHASES = {"sanity_check", "temporal_sanity_check"}

'''atomic write of run_meta.json to avoid corruption of interapted runs'''
def save_meta(meta: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = META_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    for _try in range(10):
        try:
            tmp.replace(META_PATH)
            return
        except PermissionError:
            time.sleep(0.5)
    try:
        META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.unlink(missing_ok=True)
    except OSError as e:
        print(f"warning: could not update {META_PATH.name} ({e}); the run continues")
    
'''intialise run structure "pending" and log files'''
def init_meta(run_id: str, phases: list[str]):
    return {"run_id": run_id, "started_at": datetime.now().isoformat(timespec="seconds"),"updated_at": datetime.now().isoformat(timespec="seconds"),
    "phases": {p: {"status": "pending", "started_at": None, "finished_at": None, "elapsed_s": None, "log": f"results/pipeline_{p}.log"} for p in phases}}
    
'''run a phase as subprocess and log stdout/stderr to a file'''
def run_phase(name: str, cmd: list[str], log_path: Path, env: dict | None = None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Running phase {name} at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        log_file.flush() #errors and normal output are logged to the same file
        child_env = {**os.environ, **(env or {})}
        if env:
            log_file.write("phase environment overrides: " + str(env) + chr(10))
            log_file.flush()
        process = subprocess.Popen(cmd, cwd=str(ROOT), env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout: #live progress 
            print(line, end="")
            log_file.write(line)
            log_file.flush() #flush to disk in case of crash
        return process.wait() #code 0 = success
    
'''orchestrate the full pipeline of experiments'''    
def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline of experiments.")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh run, ignoring any previous state.")
    parser.add_argument("--only", choices=[p[0] for p in PHASES + TEMPORAL_PHASES], help= "Run a single phase")
    parser.add_argument("--skip", nargs="+", choices=[p[0] for p in PHASES + TEMPORAL_PHASES], default=[], help="Skip phases")
    parser.add_argument("--smoke", action="store_true", help="Pass --smoke to run_all")
    parser.add_argument("--temporal-arm", action="store_true", dest="temporal_arm",
                        help="After the main run, repeat BoT-IoT and IoT-23 under the temporal split "
                             "into results_temporal/ and models_temporal/ (the other three datasets "
                             "have no usable timestamp, so a temporal run of them is identical)")
    args = parser.parse_args()
    if args.smoke:
        smoke_datasets() #generate smoke datasets
        os.environ["IOT_IDS_DATA_DIR"] = str(SMOKE_DATA_DIR) #environment variable for all phases to use smoke datasets
        print(f"smoke mode with {SMOKE_DATA_DIR}")
    if args.fresh:
        archive() #archive previous results
        clear_active()
    all_phases = PHASES + (TEMPORAL_PHASES if args.temporal_arm else [])
    selected = [p for p in all_phases if p[0] not in args.skip] #select phases to run
    if args.only:
        selected = [p for p in all_phases if p[0] == args.only]    
    run_id = datetime.now().strftime("pipeline_%Y%m%d_%H%M%S")
    meta = init_meta(run_id, [p[0] for p in all_phases])
    #read previous state (no reruns)
    prev_phases = {}
    if META_PATH.exists():
        try:
            prev_phases = json.loads(META_PATH.read_text(encoding="utf-8")).get("phases", {})
        except Exception:
            prev_phases = {}
    selected_names = {p[0] for p in selected}
    for name in [p[0] for p in all_phases]:
        if name not in selected_names and name in prev_phases:
            meta["phases"][name] = prev_phases[name] #keep earlier outcome
    save_meta(meta)
    #keeps a master log of the pipeline run
    master_log = RESULTS_DIR/"pipeline_run.log"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(master_log, "a", encoding="utf-8") as ml:
        ml.write(f"\nStarting pipeline run {run_id} at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    failed = False
    for name, cmd in selected:
        phase_env = dict(TEMPORAL_ENV) if name.startswith("temporal_") else None
        if name in ("run_all", "temporal_run_all") and args.smoke: #smoke mode: pass --smoke
            cmd = cmd + ["--smoke"]
        phase = meta["phases"][name]
        phase["status"] = "running"
        phase["started_at"] = datetime.now().isoformat(timespec="seconds")
        meta["updated_at"] = phase["started_at"]
        save_meta(meta)
        t0 = time.time()
        log_path = RESULTS_DIR/f"pipeline_{name}.log"
        rc = run_phase(name, cmd, log_path, env=phase_env)
        elapsed = round(time.time() - t0, 1)
        phase["finished_at"] = datetime.now().isoformat(timespec="seconds")
        phase["elapsed_s"] = elapsed
        phase["status"] = "done" if rc == 0 else ("findings" if name in ADVISORY_PHASES else "failed")
        save_meta(meta)
        with open(master_log, "a", encoding="utf-8") as ml:
            ml.write(f"Phase {name} finished at {datetime.now():%Y-%m-%d %H:%M:%S} with return code {rc}, elapsed time: {elapsed}s\n")
        if rc != 0 and name in ADVISORY_PHASES:
            print(f"Phase {name} reported findings (exit {rc}); see {log_path.name}.")
        elif rc != 0:
            failed = True
            if name in CRITICAL_PHASES:
                print(f"Critical phase {name} failed with return code {rc}. Aborting pipeline.")
                break #critical phase failed, abort the pipeline
            print(f"Phase {name} failed with return code {rc}. Continuing with next phases.")        
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["status"] = "done" if not failed else "failed"
    save_meta(meta)
    if failed:
        sys.exit(1) #non-zero exit code to indicate failure
    print(f"\nPipeline run {run_id} completed successfully at {datetime.now():%Y-%m-%d %H:%M:%S}.")    
    print("open dashboard: streamlit run src/trainingmonitor.py")


    
if __name__ == "__main__":
    main()
    