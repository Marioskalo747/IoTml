#Libraries
import argparse
import json
import os
import shutil
import stat
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ARCHIVES_DIR, MODELS_DIR, PROJECT_DIR, RESULTS_DIR


'''deletion error handler for read-only files'''
def on_rm_error(func, path, exc):
    for i in range(4): 
        try:
            os.chmod(path, stat.S_IWRITE) #make writable
            func(path) #re-run
            return
        except Exception:
            time.sleep(0.4) #bries wait
    print(f"  [warn] could not delete (locked/in use): {path}") 
    


'''directory removal for locked files'''
def force_rmtree(path: Path):
    if not path.exists():
        return #nothing to delete
    shutil.rmtree(path, onexc=on_rm_error) #onerror handler for read-only files

'''timestamp for archive naming'''    
def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

'''recursive file count'''
def count_files(root: Path):
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())

'''sibling directory the temporal arm writes into, derived exactly as runfull.TEMPORAL_ENV does'''
def temporala(d: Path):
    return d.with_name(d.name + "_temporal")

'''archive current results and models directories'''
def archive(label: str | None = None):
    stamps = label or stamp()
    dest = ARCHIVES_DIR / f"run_{stamps}"
    if dest.exists():
        raise SystemExit(f"Archive already exists: {dest}") #never overwrite existing archive

    dest.mkdir(parents=True)
    manifest = {"run_id": f"run_{stamps}", "archived_at": datetime.now().isoformat(timespec="seconds"), 
                "project_dir": str(PROJECT_DIR), "results_files": 0, "models_files": 0} #run's identity card
    for src, name in [(RESULTS_DIR, "results"), (MODELS_DIR, "models"),
                      (temporala(RESULTS_DIR), "results_temporal"), (temporala(MODELS_DIR), "models_temporal")]:
        if src.exists() and any(src.iterdir()): #empty folders are not archived
            shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.tmp"))
            manifest[f"{name}_files"] = count_files(dest / name)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Archived -> {dest}")
    print(f"  results: {manifest['results_files']} files, models: {manifest['models_files']} files")
    return dest

'''print available archives'''
def list_archives():
    if not ARCHIVES_DIR.exists():
        return []
    runs = sorted(ARCHIVES_DIR.glob("run_*"), reverse=True) #names are timestamps
    for run in runs:
        manifest_path = run / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"  {manifest['run_id']}: ({manifest.get('archived_at', '?')})" f" results: {manifest.get('results_files', '?')} files, models: {manifest.get('models_files', '?')}")
        else:
            print(f"  [warn] missing manifest: {manifest_path}")
    if not runs:
        print("  (no archives found)")
    return runs

'''restore a results from old archive (backup current results/models first)'''
def restore(run_id: str):
    run_dir = ARCHIVES_DIR / run_id
    if not run_dir.exists():
        raise SystemExit(f"Archive not found: {run_dir}") #safe guard
    if  RESULTS_DIR.exists() or MODELS_DIR.exists():
        backup = archive(label=f"pre_restore_{run_id}_{stamp()}")
        print(f"Backed up current results/models to: {backup.name}")
    for name in ("results", "models"):
        active = PROJECT_DIR / name
        archived = run_dir / name
        if active.exists():
            force_rmtree(active) #clean replacement
        if archived.exists():
            shutil.copytree(archived, active)
            print(f"Restored {name} from archive: {run_id}")
    
'''wipe results completely'''    
def clear_active():
    for d in (RESULTS_DIR, MODELS_DIR):
        if d.exists():
            force_rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for d in (temporala(RESULTS_DIR), temporala(MODELS_DIR)):
        if d.exists():
            force_rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    (temporala(RESULTS_DIR) / "figures").mkdir(parents=True, exist_ok=True)
    print("Cleared active results/, models/ and their _temporal counterparts (figures dir recreated).")

'''CLI archive, list, restore, clear'''    
def main():
    parser = argparse.ArgumentParser(description="Archive/restore results and models directories.")
    parser.add_argument("--list", action="store_true", help="List available archives.")
    parser.add_argument("--restore", metavar="RUN_ID", help="Restore a specific archive by its run ID." )
    parser.add_argument("--fresh", action="store_true", help="Clear active results and models directories.")  ####ALLAGH 
    parser.add_argument("--label", help="Label for the archive (default: timestamp).")
    args = parser.parse_args()
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True) #create if missing
    if args.list:
        list_archives()
        return
    if args.restore:
        restore(args.restore)
        return
    if args.fresh:
        archive(label=args.label) #archive current results/models before clearing
        clear_active()
        print("\nReady for fresh run:")
        print("  python src/runfull.py")
        print("  python src/runall.py")
        return
    archive(label=args.label) #plain archive

if __name__ == "__main__":
    main()
    
    
        
        