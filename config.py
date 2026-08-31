#Libraries
from pathlib import Path
import os
import sys

#stdout and stderr reconfiguration to avoid encoding issues
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PROJECT_DIR = Path(__file__).parent

DATASET_DIR_CANDIDATES= [Path(r"C:\Users\kalou\Downloads\κωδικας3\datasets"), PROJECT_DIR.parent / "datasets", PROJECT_DIR / "datasets"]    ########PATHS

#pick datasets (first candidate that exists)
def dataset_dir():
    override = os.environ.get("IOT_IDS_DATA_DIR") #env-var for --smoke 
    if override:
        return Path(override)
    for d in DATASET_DIR_CANDIDATES:
        if d.exists():
            return d
    return PROJECT_DIR / "datasets"

#path configurations.

MODELS_DIR = PROJECT_DIR / os.environ.get("IOT_IDS_MODELS_DIR", "models")
RESULTS_DIR = PROJECT_DIR / os.environ.get("IOT_IDS_RESULTS_DIR", "results")
FIGURES_DIR = RESULTS_DIR / "figures" 
ARCHIVES_DIR = PROJECT_DIR / os.environ.get("IOT_IDS_ARCHIVES_DIR", "archives")


MAX_ROWS_PER_DATASET = 400000 #loaded rows per dataset
MAX_TRAIN_ROWS = 250000  #training rows after the split
#cap for slow models (KNN, LinearSVM, MLP, DeepMLP, TorchMLP, AdaBoost). They therefore see ~4x
#fewer rows than the boosting models, which confounds any "boosting beats neural networks" claim.
#Set IOT_IDS_SLOW_CAP=250000 to run the matched-budget comparison (much slower, KNN especially)
SLOW_MODEL_TRAIN_CAP = int(os.environ.get("IOT_IDS_SLOW_CAP", 60000))
TEST_SIZE = 0.30  #train-test split
RANDOM_STATE = 42  #random state for reproducibility
MIN_CLASS_ROWS = 60 #classes with less than this merged into "Other"

#Per-class floor when dataset is capped down to MAX_ROWS_PER_DATASET
MIN_ROWS_PER_CLASS_LOAD = int(os.environ.get("IOT_IDS_MIN_CLASS_LOAD", 8000))

#Same idea for the slow-model subsample
MIN_ROWS_PER_CLASS_SLOW = int(os.environ.get("IOT_IDS_MIN_CLASS_SLOW", 2000))

#Split protocol. random is the stratified split used throughout the literature
SPLIT_MODE = os.environ.get("IOT_IDS_SPLIT", "random").strip().lower()
TS_COL = "__ts"  #reserved column carrying capture time; split() removes it before any model sees it
SPLIT_AUDIT_PATH = RESULTS_DIR / "split_audit.json" #records which protocol each dataset actually got

#Heavy-tailed features
LOG_SCALE_FEATURES = os.environ.get("IOT_IDS_LOG_SCALE", "1") not in ("0", "false", "False")

#Cost-sensitive learning: how many false alarms one missed attack is worth
FN_COST_RATIO = float(os.environ.get("IOT_IDS_FN_COST", 1.0))

#Shared features and taxonomy
UNIFIED_CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Botnet", "Recon", "BruteForce", "Web", "Spoofing_MITM", "Injection", "Ransomware", "Theft", "Other"]
COMMON_FEATURES = ["duration", "total_pkts", "total_bytes", "pkt_rate", "byte_rate", "avg_pkt_size", "proto_tcp", "proto_udp", "proto_icmp", "dst_port", "src_port"]


