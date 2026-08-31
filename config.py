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
#Both are redirectable through the environment so a SECOND split protocol can run inside the same
#pipeline without writing over the first one's results. Unset, the behaviour is exactly as before
MODELS_DIR = PROJECT_DIR / os.environ.get("IOT_IDS_MODELS_DIR", "models")
RESULTS_DIR = PROJECT_DIR / os.environ.get("IOT_IDS_RESULTS_DIR", "results")
FIGURES_DIR = RESULTS_DIR / "figures" 
#redirectable for the same reason as the two above: a smoke run must not archive its synthetic
#results next to the real ones, and sanitycheck must not compare a smoke run against a real archived
#run (it did, and reported a page of meaningless "large change" lines)
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

#Per-class floor when a dataset is capped down to MAX_ROWS_PER_DATASET. A proportional cap alone
#shrinks a rare class by the same factor as DDoS, so CICIoT2023 BruteForce arrived with 1,500 of the
#~7,600 rows the sampler had already read. Macro-F1 weights that class exactly as much as DDoS, so
#starving it is the single cheapest way to lose points. The floor keeps small classes whole; large
#classes are still reduced proportionally, so the total grows only a little
MIN_ROWS_PER_CLASS_LOAD = int(os.environ.get("IOT_IDS_MIN_CLASS_LOAD", 8000))

#Same idea for the slow-model subsample. That path had NO floor at all: a class was scaled by
#cap/len(train) whatever its size, so IoT-23 DDoS went to 65 rows and BoT-IoT Benign to 113. The
#slow models were not merely trained on fewer rows, they were trained on almost none of the rare
#classes they are then scored on
MIN_ROWS_PER_CLASS_SLOW = int(os.environ.get("IOT_IDS_MIN_CLASS_SLOW", 2000))

#Split protocol. "random" is the stratified split used throughout the literature; "temporal" sorts
#by capture time and puts the OLDEST 70% in train, the NEWEST 30% in test, so flows from the same
#attack burst cannot land on both sides (Arp et al., USENIX Sec 2022: "temporal snooping").
#Random splits are known to overestimate NIDS performance, so both protocols should be reported.
#Datasets without a usable timestamp fall back to "random" and are flagged in split_audit.json.
SPLIT_MODE = os.environ.get("IOT_IDS_SPLIT", "random").strip().lower()
TS_COL = "__ts"  #reserved column carrying capture time; split() removes it before any model sees it
SPLIT_AUDIT_PATH = RESULTS_DIR / "split_audit.json" #records which protocol each dataset actually got

#Heavy-tailed features. total_bytes reaches 3.9e9, byte_rate 1.9e8, and after StandardScaler the
#largest z-score in a column runs from 100 to 632 depending on the dataset: a handful of flows own
#the whole range and every ordinary flow is squashed into a sliver around zero. Trees are invariant
#to this, linear models, SVMs, KNN and the neural nets are not, which is a preprocessing artefact
#sitting inside the "boosting beats linear" comparison. A signed log1p is monotone (it cannot
#reorder anything), defined at zero and on the -1 "unknown port" sentinel, and needs no fitting, so
#it carries no leakage risk. Set IOT_IDS_LOG_SCALE=0 to reproduce the untransformed behaviour.
LOG_SCALE_FEATURES = os.environ.get("IOT_IDS_LOG_SCALE", "1") not in ("0", "false", "False")

#Cost-sensitive learning: how many false alarms one missed attack is worth. 1.0 = current behaviour
#(no weighting). Higher values raise recall on the rare attack classes at the cost of precision
FN_COST_RATIO = float(os.environ.get("IOT_IDS_FN_COST", 1.0))

#Shared features and taxonomy
#"Other" is the bucket merge_rare_classes drops small classes into, so no NAMED attack family may
#share it: TON-IoT ransomware used to land there, which made "Other" mean a specific malware family
#in one dataset and "assorted leftovers" in another - two different things under one label in the
#pooled model and in every LODO comparison
UNIFIED_CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Botnet", "Recon", "BruteForce", "Web", "Spoofing_MITM", "Injection", "Ransomware", "Theft", "Other"]
COMMON_FEATURES = ["duration", "total_pkts", "total_bytes", "pkt_rate", "byte_rate", "avg_pkt_size", "proto_tcp", "proto_udp", "proto_icmp", "dst_port", "src_port"]


