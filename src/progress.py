#Libraries
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR

#Stage names for UI phases
PHASE_STAGES = {"run_all": ["loading", "training", "LODO", "finished"],
    "tuning": ["loading", "tuning", "refit", "finished"],
    "extended_study": ["loading", "imbalance", "feature_selection", "learning_curve", "cv", "finished"],
    "deep_study": ["loading", "ablation", "learning_curve", "tuning", "finished"],
    "anomaly_study": ["loading", "in_domain", "cross_domain", "finished"],
    "adaptation_study": ["loading", "adaptation", "finished"],
    "deployment_bench": ["loading", "benchmark", "finished"],
    "extended_figures": ["loading", "backfill", "heatmaps", "curves", "shap", "finished"],
    "report_assets": ["loading", "tables", "figures", "finished"]}

'''Tracker with live state'''
class Progress:
    '''initialization'''
    def __init__(self, total_steps: int, phase: str = ""):
        self.path = RESULTS_DIR / "progress.json" #state file
        self.t0 = time.time() #start time
        self.total = max(total_steps, 1)
        self.done = 0
        self.phase = phase  #phases
        self.stage = "starting"  #main step of the phase
        self.substage = ""
        self.dataset = ""
        self.task = ""
        self.model = ""
        self.message = ""
        self.details: dict = {}  #extra info
        self.recent: list[str] = []
        self.write()

    '''fields update'''
    def update(self, stage=None, dataset=None, task=None, model=None,
               message=None, details=None, substage=None):
        if stage is not None:
            self.stage = stage
        if substage is not None:
            self.substage = substage
        if dataset is not None:
            self.dataset = dataset
        if task is not None:
            self.task = task
        if model is not None:
            self.model = model
        if message is not None:
            self.message = message
            self.log(message)  #message is logged to history
        if details is not None:
            mergedet(self.details, details)
        self.write()

    '''advance progress by n steps'''
    def advance(self, n: int = 1, **kw):
        if kw:
            self.update(**kw)
        self.done += n
        self.write()

    '''timestamp log message (last 50)'''
    def log(self, msg: str):
        self.recent.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.recent = self.recent[-50:]
        self.write()

    '''finish phase (100% counter)'''
    def finish(self, message="Finished"):
        self.stage = "finished"
        self.substage = ""
        self.done = self.total #100% bar
        self.message = message
        self.log(message)
        self.write()

    '''elapsed time, ETA, percentage, and JSON file'''
    def write(self):
        elapsed = time.time() - self.t0
        eta = (elapsed / self.done*(self.total - self.done)) if self.done else None
        state = {"phase": self.phase, "stage": self.stage, "substage": self.substage,
        "stages": PHASE_STAGES.get(self.phase, []),
        "dataset": self.dataset, "task": self.task, "model": self.model,
        "message": self.message, "details": self.details, "done": self.done, "total": self.total,
        "pct": round(100 * self.done / self.total, 1), "elapsed_s": round(elapsed, 1),
        "eta_s": round(eta, 1) if eta is not None else None, "updated": time.time(), "recent": self.recent}
        #tmp file to avoid partial writes then os.replace
        tmp = self.path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent = 1)
            os.replace(tmp, self.path)
        except OSError:
            pass

'''Deep recursive merge'''
def mergedet(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k].update(v) #keep existing keys
        else:
            dst[k] = v
