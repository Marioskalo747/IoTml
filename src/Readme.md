# IoT Intrusion Detection — Experimental Pipeline

Comparative study of machine learning models for intrusion detection in IoT networks, on **4 public datasets** (BoT-IoT, ToN-IoT, CICIoT2023, IoT-23), with **two
split protocols** (random / temporal), **19 models**, two tasks
(binary and multiclass per attack category) and **18** experiment
phases that produce 186 result records, 33 CSV tables and 402 figures.

A full run takes about **20 hours** on a laptop with 16 GB RAM and CUDA.
The last recorded run (`results/run_meta.json`) finished in
**17 hours 31 minutes** (15 h 35 m the main arm + 1 h 55 m the temporal one).

#Table of contents

1. [Quick start](#quick-start)
2. [System requirements](#system-requirements)
3. [Folder structure](#folder-structure)
4. [Datasets and where they go](#datasets--where-they-go)
5. [How the pipeline runs](#how-the-pipeline-runs)
6. [Commands & flags](#commands--flags)
7. [Streamlit monitor](#streamlit-monitor)
8. [Run times](#run-times)
9. [CUDA / GPU (what needs it and what does not)](#cuda--gpu--what-needs-it-and-what-does-not)
10. [Temporal split (chronological split)](#temporal-split--the-chronological-split)
11. [Where the results are stored](#where-the-results-are-stored)
12. [Environment variables](#environment-variables)
13. [Resume, checkpointing & archiving](#resume-checkpointing--archiving)
14. [File map](#file-map)
15. [Reproducibility](#reproducibility)
16. [Troubleshooting](#troubleshooting)


# Quick start

**0.** All commands
run **from the parent folder**. Details:
[Folder structure](#folder-structure--where-src-must-sit).

```
project/
├── src/          <- the code
└── datasets/     <- the data
```

```bash
pip install -r src/requirements.txt
```

[`requirements.txt`](requirements.txt) pins the **exact versions** with which all
the results were produced. For the CUDA build of PyTorch (optional, if absent everything runs on the CPU):
`pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`.

**1. Test without data** (synthetic datasets, writes to `results_smoke/`):

```bash
python src/runfull.py --smoke
```

**2. Full run** (needs the real datasets, see [Datasets](#datasets--where-they-go)):

```bash
python src/runfull.py --fresh --temporal-arm
```

**3. Monitoring in another terminal**:

```bash
streamlit run src/trainingmonitor.py
```

# System requirements

## Software

| | Tested version | Note |
|---|---|---|
| Python | **3.14.0** (Windows 11, 64-bit) | also works on 3.11+ |
| numpy, pandas, scipy | 2.3.5 / 2.3.3 / 1.16.3 | |
| scikit-learn | 1.7.2 | the core of all pipelines |
| imbalanced-learn | 0.14.0 | SMOTE / over- / under-sampling |
| xgboost, lightgbm, catboost | 3.1.3 / 4.6.0 / 1.2.10 | **optional** if missing, `model_zoo` simply does not add them |
| torch | 2.9.1+cu128 | **optional** only for the `TorchMLP` model |
| optuna | 4.9.0 | `tuning` phase (TPE sampler) |
| matplotlib, seaborn, plotly | 3.10.7 / 0.13.2 / 6.5.0 | PNG figures + interactive charts in the monitor |
| streamlit | 1.58.0 | dashboard |
| joblib | 1.5.2 | model serialisation (`compress=3`) |

All versions in the table are in [`requirements.txt`](requirements.txt).

The optional libraries are detected with `try/except ImportError` in
`models_zoo.py`. The pipeline also runs without them, just with fewer models.

## Hardware

| Resource | Minimum | Recommended (reference machine) |
|---|---|---|
| **RAM** | 8 GB (with reduced limits) | **16 GB** |
| CPU | 4 cores | 8+ cores (most models run with `n_jobs=-1`) |
| **GPU/CUDA** | **optional** | NVIDIA RTX 4050 Laptop (6 GB VRAM), CUDA 12.8 |
| Disk | 5 GB free | 10 GB (models 1.3 GB + results 146 MB + one archive per run) |

**Why 16 GB RAM:** the load cap is `MAX_ROWS_PER_DATASET = 400,000`
rows per dataset and `MAX_TRAIN_ROWS = 250,000` training rows after the
split ([`config.py`](config.py)). The `combined` dataset reaches 1.08 million rows.
The peak occurs in `run_all` (loading + preprocessing + `RandomForest`/`ExtraTrees`
with 200 trees) and in `seed_study` (4 seeds × 15 models).

`train_all_models()` catches `MemoryError` per model: if a model does not fit, it is skipped, `gc.collect()` is called and the pipeline continues instead of crashing.

**If you have less RAM**, lower the limits before the run:

```bash
IOT_IDS_SLOW_CAP=30000 IOT_IDS_MIN_CLASS_LOAD=4000 python src/runfull.py
```

or edit `MAX_ROWS_PER_DATASET` / `MAX_TRAIN_ROWS` directly in [`config.py`](config.py).

# Folder structure

The system in the parent folder consists of `src/` and `datasets/` and the subfolders of src with the `results_*/`

```
<IoTml>/           <- the parent folder (project root)
│
|── src/                        
│   |── config.py  runall.py  runfull.py  loaders.py  …
│   |── results/                <- created automatically on the first run
│   |── models/                 <- created automatically
│   └── archives/               <- created automatically
│
|── datasets/                   <- the datasets folder
│   |── bot_iot.zip
│   |── ciciot2023.zip
│   |── iot_23.tar.gz
│   └── ton_iot.csv
│
|── .streamlit/config.toml      <- optional (dashboard settings)
|── datasets_config.json        <- optional (only for custom datasets)
|── status.py                   <- optional (status as text)
|── smoke_data/                 <- created by --smoke
└── datasets/local_cache/      <- created on its own if the datasets are on another drive
```

##The folder must be named `src`

[`runfull.py`](runfull.py) builds the phase paths as
`<parent>/src/<script>.py` (`ROOT = Path(__file__).resolve().parents[1]`).

The **other** commands (`runall.py`, `tuning.py`, all the studies) work with
any folder name, because they use `Path(__file__).parent`. **Only
`runfull.py` has this requirement** — and it is the main command, so keep the name `src`.

## What needs changing in files

**In the normal case, nothing.** The only reason to touch [`config.py`](config.py)
is the hard-coded data path of the development machine:

```python
DATASET_DIR_CANDIDATES = [Path(r"path/to/datasets/in/files"), 
PROJECT_DIR.parent / "datasets",     # <parent>/datasets
PROJECT_DIR / "datasets"]            # src/datasets
```

It returns the first folder that **exists**.
You can replace the first line with your own path to the datasets.

Three ways to declare where the data is, from the simplest:

| # | Way | When |
|---|---|---|
| 1 | Create `datasets/` **next to `src/`** | the recommended solution, no code change |
| 2 | Create `datasets/` **inside `src/`** | if you only have the `src` folder and want nothing outside it |
| 3 | Set `IOT_IDS_DATA_DIR` | if the data is on another drive or a shared folder |

```bash
IOT_IDS_DATA_DIR=/mnt/data/iot_datasets python src/runfull.py
```

```powershell
$env:IOT_IDS_DATA_DIR="D:\iot_datasets"; python src\runfull.py
```

`IOT_IDS_DATA_DIR` **overrides all** the candidates, so it is also the way to bypass the hard-coded path without touching the code.

## Verification before the run

```bash
python -c "import sys; sys.path.insert(0,'src'); from config import dataset_dir, PROJECT_DIR, RESULTS_DIR; from loaders import available_datasets; print('code   :', PROJECT_DIR); print('data   :', dataset_dir()); print('results:', RESULTS_DIR); print('found  :', available_datasets())"
```

Correct output:

```
code   : ...\project\src
data   : ...\project\datasets
results: ...\project\src\results
found  : ['bot_iot', 'ton_iot', 'ciciot2023', 'iot23']
```

If `found` is an empty list, the files do not have the names the code expects.


# Datasets

## Official sources and files used

| Dataset | Official page | What was used | Name in `datasets/` | What the loader reads | Publication to cite |
|---|---|---|---|---|---|
| **IoT-23** | <https://www.stratosphereips.org/datasets-iot23> (and Zenodo, [doi:10.5281/zenodo.4743746](https://doi.org/10.5281/zenodo.4743746)) | the light version of the archive: only the Zeek logs `conn.log.labeled`, without the pcaps | `iot_23.tar.gz` (9.37 GB) | all `opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-*/bro/conn.log.labeled`, up to 60,000 flows per scenario | S. Garcia, A. Parmisano, M. J. Erquiaga, *IoT-23: A labeled dataset with malicious and benign IoT network traffic*, Stratosphere Lab, 2020 |
| **BoT-IoT** | <https://research.unsw.edu.au/projects/bot-iot-dataset> | the `5%` folder of the distribution (official 5% subsample), compressed into a zip | `bot_iot.zip` (1.99 GB) | **only** the `5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`· the `10-best features` are ignored | N. Koroniotis et al., *Future Generation Computer Systems*, vol. 100, 2019 |
| **ToN-IoT** | <https://research.unsw.edu.au/projects/toniot-datasets> | the network traffic CSV of the train/test sets, `Train_Test_Network.csv` (211,043 flows, columns `label` and `type`) | `ton_iot.csv` (29.9 MB) | all rows· label from the `type` column | N. Moustafa, *Sustainable Cities and Society*, vol. 72, 2021· A. Alsaedi et al., *IEEE Access*, vol. 8, 2020 |
| **CICIoT2023** | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | the feature CSVs per capture (`<Attack>/<name>.pcap.csv`, 34 folders: 33 attacks + Benign), compressed into a zip with root `CSV/` | `ciciot2023.zip` (1.43 GB) | all `CSV/<Attack>/*.pcap.csv`, with a row budget **per class** (`ciciot_budget`)· label from the `label` column if present, otherwise from the folder name | E. C. P. Neto et al., *Sensors*, vol. 23, no. 13, 2023 |

The full bibliographic references and the description of each set are in the thesis documentation.
Using each set requires citing the publications its source asks for, and the terms of
use and redistribution must be checked on the official page.

## The layout that works

Download the 4 datasets from their official pages, **rename** the files and
leave them **as they are, compressed**, in a `datasets/` folder:

```
datasets/
├── bot_iot.zip        1.99 GB   BoT-IoT        — UNSW Canberra
├── ciciot2023.zip     1.43 GB   CICIoT2023     — Canadian Institute for Cybersecurity (UNB)
├── iot_23.tar.gz      9.37 GB   IoT-23         — Stratosphere Laboratory (CTU)
└── ton_iot.csv       29.9 MB    ToN-IoT        — UNSW Canberra
                     ─────────
                      12.8 GB in total
```

This is exactly the structure of the development machine and it is the **simplest that
satisfies all the patterns**. No decompression is needed, the loaders read by
streaming directly from inside the zip / tar.gz, which is why you do not need 50 GB of free space for decompression.

## What each loader accepts

The search is **recursive** (`rglob`), so the files can also be in
subfolders of `datasets/`, as long as the **name** matches one of the patterns
(`BUILTIN_PROBES` in [`loaders.py`](loaders.py)):

| Dataset | Recognised names | What it must contain |
|---|---|---|
| **bot_iot** | `bot_iot*.zip` `5%*.zip` `*5pc*.zip` `*bot*iot*.zip` `*Full5pc*.csv` | The CSVs of the **5% subset**: `5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`. The loader keeps **only** the members that contain `Full5pc` the `10-best features` are ignored |
| **ton_iot** | `ton_iot*.csv` `Train_Test_Network*.csv` `train_test_network*.csv` | The single CSV `Train_Test_Network.csv` (`type` column for the label). It is already balanced: 8 classes × 20,000 |
| **ciciot2023** | `ciciot2023*.zip` `CSV.zip` `*CICIoT*.zip` `*ciciot*.zip` `*.pcap.csv` | The per-attack CSVs: `CSV/<Attack_Name>/<Attack_Name>.pcap.csv`. The label is read from the `label` column if present, otherwise from the folder name |
| **iot23** | `iot_23*.tar.gz` `*iot*23*.tar.gz` `conn.log.labeled` `*.log.labeled` | The Zeek logs `.../IoTScenarios/CTU-IoT-Malware-Capture-<N>-<M>/bro/conn.log.labeled` |

**Two alternatives if you have already decompressed:**

* CICIoT2023: leave the `*.pcap.csv` inside their per-attack folders and the loader finds them.
* IoT-23: leave the `conn.log.labeled` in their folder structure.

**You do not need all 4.** `available_datasets()` returns whatever it found and the pipeline
runs with those. With fewer than 2 datasets, however, `combined` and `LODO` are skipped
automatically (`not enough datasets for combined model`).

## Datasets on another drive

If `IOT_IDS_DATA_DIR` points to a **different drive letter** from the code,
`localize()` first copies the file to `<parent>/datasets/local_cache/` — with
a free-space check (+20% margin) and 3 attempts. Streaming the
compressed files from an external or network drive is much slower than copying.

> Mind the space: the cache is **always** built next to the code, not on the data drive. For `iot_23.tar.gz` 11 GB must be free there.

## Adding your own dataset

Unlike the builtin ones, a custom dataset **must** be in its own subfolder
of `datasets/`, and the folder name becomes the dataset name:

```
datasets/
└── myset/                  <- the dataset name
    ├── part1.csv
    └── part2.csv
```

To enable it, create a file **`<parent>/datasets_config.json`** **next to `src/`, not inside it**:

```json
{ "custom": { "myset": { "enabled": true, "label_col": "attack_type", "glob": "*.csv" } } }
```

`label_col` is optional (if missing, `detect_label_col()` looks for the usual
names `label`, `attack`, `attack_type`, `attack_cat`, `category`, `type`, `class`, …).
`generic_loader()` takes care of unifying the taxonomy (`unify_label`) and extracting
the 11 common features. Enabling is **explicit (opt-in)**:
`discover_candidates()` sees the folder, but without `"enabled": true` nothing gets into
the experiments by accident.

## Preprocessing

All preprocessing is done by the code, **every time from the raw files**. No intermediate "cleaned" file that has to be made by hand is needed. The steps, in order:

**1. Loader per dataset** ([`loaders.py`](loaders.py)):

| Dataset | Label | What is removed | What changes |
|---|---|---|---|
| BoT-IoT | `category` `BOTIOT_LABEL_MAP` (normal, Benign, DDoS, DoS, reconnaissance, Recon, theft, Theft) | `pkSeqID`, `stime`, `ltime`, `seq`, `saddr`, `daddr` and the label columns `attack`, `category`, `subcategory` (`BOTIOT_DROP`) | 12 context features (5 and 10 s windows) from IP and time, before they are removed (hex ports to decimal (`parse_port`)) |
| ToN-IoT | `type` `TONIOT_LABEL_MAP` (10 classes), the original binary `label` column is replaced by the unified label | IPs, `ts` and high-cardinality text fields: `dns_query`, `http_uri`, `http_user_agent`, `ssl_subject`, `ssl_issuer`, `weird_*` etc. (`TONIOT_DROP`) | Zeek's dash - becomes NaN |
| CICIoT2023 | folder name (or `label` column) `ciciot_map_label` (10 classes) | it has no IPs or ports | reading in chunks of 250,000 rows with a per-class budget |
| IoT-23 | `label` + `detailed-label` `iot23_map_label` (4 classes) | IPs, `uid`, `ts`, `history`, `local_orig`, `local_resp`, `tunnel_parents`, `detailed-label` | 12 context features over consecutive row blocks (`_windowed_blocks`) |

In all four: classes with fewer than 60 flows are merged into `Other` (`merge_rare_classes`),
and each set is capped, stratified, at 400,000 flows with a minimum of 8,000 per class
(`stratified_cap`). The timestamp is kept only in the `ts` column, for the chronological split.

**2. Common schema of 11 features** (the `*_common` functions in [`loaders.py`](loaders.py)):
`duration`, `total_pkts`, `total_bytes`, `pkt_rate`, `byte_rate`, `avg_pkt_size`,
`proto_tcp/udp/icmp`, `dst_port`, `src_port`. Rate 0 for zero-duration flows. In CICIoT2023
the ports are 0 and the packet rate is the native `rate` field. It is stored in
`results/common_cache/<dataset>.pkl` and feeds the union (`combined`), LODO and the
transfer, adaptation and anomaly studies.

**3. Split** (`split()` in [`preprocessing.py`](preprocessing.py)):

1. removal of exact duplicates (flow columns + label) **before** the split.
2. stratified 70/30 split with seed 42 (or chronological, with `IOT_IDS_SPLIT=temporal`).
3. training cap of 250,000 rows, proportional per class with a minimum of 1,000, without capping the test.
4. the 30 most frequent values of each categorical column are kept, the rest become `RARE`.
5. removal of columns with a single value in train.

The slow models (k-NN, LinearSVM, AdaBoost, MLP, DeepMLP, TorchMLP) are trained on a stratified
subsample of 60,000 rows, with a minimum of 2,000 per class.

**4. Transformer per model** (`make_preprocessor()`, inside a `Pipeline`, `fit` only on train):
numeric-> median imputation + missingness indicators-> `sign(x)·log1p(|x|)`-> `StandardScaler` and
categorical -> most-frequent + one-hot. Each port column gives an extra 19 indicators (4 ranges and 15
services: 21, 22, 23, 25, 53, 80, 123, 443, 445, 502, 1883, 3389, 5683, 8080, 8883).

## Processed subsets and terms of use

The repository **contains no data**: `datasets/`, `results/` and all `*.pkl` are on the drive. The pipeline produces one processed subset per dataset,
`results/common_cache/<dataset>.pkl`: the 11 common features together with the columns `label`, `binary`
and `source`, after the loaders' sampling. These subsets **are not
published**, for two reasons:

1. **Terms of use.** They are derivatives of the original sets, and each source sets its own terms
   of use and redistribution.
2. **They are not enough for reproduction.** They contain only the common schema. The per-dataset models use
   the native features, which exist only in the raw files. From the cache alone only
   the common-schema studies could be rerun (`domain_shift_study`, `adaptation_study`,
   `anomaly_study`, `anomaly_sweep`).

Full reproduction is done from the official files with the commands of
[§ Quick start](#quick-start). The chain is deterministic: the raw files are reloaded and exactly the same splits and metrics are reproduced.

# How the pipeline runs

[`runfull.py`](runfull.py) is the **orchestrator**. It runs each phase as a **separate
process** (`subprocess.Popen`), so that a crash or memory leak in one phase does not
drag down the others. Each phase writes its own log
(`results/pipeline_<phase>.log`) and the status of all of them is updated atomically in
`results/run_meta.json`.

## Main arm of 18 phases

| # | Phase | Script | What it does |
|---|---|---|---|
| 1 | `run_all` | [`runall.py`](runall.py) | **Critical phase.** Loading -> preprocessing -> split -> training 15 models × 2 tasks × 4 datasets, + combined model, + LODO cross-dataset |
| 2 | `tuning` | [`tuning.py`](tuning.py) | Optuna TPE, 25 trials × 3-fold CV × 30,000 rows, for RF/XGB/LGBM/CatBoost |
| 3 | `extended_study` | [`extendedstudy.py`](extendedstudy.py) | A: imbalance strategies B: feature selection C: sample size D: cross-validation |
| 4 | `deep_study` | [`deepstudy.py`](deepstudy.py) | Imbalance ablation, learning curves, RandomizedSearchCV |
| 5 | `anomaly_study` | [`anomalystudy.py`](anomalystudy.py) | IsolationForest trained only on benign (in-domain + cross-domain) |
| 6 | `adaptation_study` | [`adaptionstudy.py`](adaptionstudy.py) | How much local data is needed (0%, 0.5%, 1%, 2%, 5%, 10%) to adapt to a new network |
| 7 | `deployment_bench` | [`deploymentbench.py`](deploymentbench.py) | Latency/throughput: 50,000 flows × 3 repetitions per model, model size on disk |
| 8 | `extended_figures` | [`make_extended_figures.py`](make_extended_figures.py) | ROC/PR curves, heatmaps, SHAP |
| 9 | `report_assets` | [`reportassets.py`](reportassets.py) | Produces the `tables.csv` and the summary figures |
| 10 | `validation_stats` | [`validationstats.py`](validationstats.py) | Bootstrap 95% CI (500 samples), McNemar tests, dummy baselines |
| 11 | `domain_shift_study` | [`domainshift_study.py`](domainshift_study.py) | How recognisable the source dataset is (source identifiability) |
| 12 | `anomaly_sweep` | [`anomalysweep.py`](anomalysweep.py) | Contamination sweep (0.5% → 10%) × FPR targets |
| 13 | `leakage_ablation` | [`leakage_ablation.py`](leakage_ablation.py) | Removal of ports / per-IP aggregates / windowed features — how much of the performance was leakage |
| 14 | `operating_points` | [`operatingpoints.py`](operatingpoints.py) | Base-rate-corrected precision, alert volume on a network of 10M flows/day |
| 15 | `cost_sensitive` | [`costsensitive.py`](costsensitive.py) | `none` / `balanced` / `fn_cost` × costs 1×, 5×, 10× |
| 16 | `seed_study` | [`seedstudy.py`](seedstudy.py) | Stability across 4 seeds (42, 7, 13, 34) |
| 17 | `specialist_study` | [`specialiststudy.py`](specialiststudy.py) | A: cascade (gate + specialist) B: one-vs-rest specialists C: per-class cross-dataset transfer |
| 18 | `sanity_check` | [`sanitycheck.py`](sanitycheck.py) | Advisory (checks out-of-range metrics, stalls, inconsistencies) |

## Temporal arm (with `--temporal-arm`)

It runs **only for bot_iot and iot23** (the only datasets with a usable timestamp)
and writes to separate folders so as not to overwrite the main arm:
`temporal_run_all -> temporal_report_assets -> temporal_validation_stats ->
temporal_seed_study -> temporal_sanity_check`.

## Error policy

| Category | Phases | Behaviour on exit code ≠ 0 |
|---|---|---|
| **Critical** | `run_all` | the pipeline **stops** |
| **Advisory** | `sanity_check`, `temporal_sanity_check` | reported as `findings`, not treated as a failure |
| Others | all the rest | the failure is recorded, the pipeline **continues** to the next phase |

# Commands & flags

## `runfull.py`

```bash
python src/runfull.py [--fresh] [--only PHASE] [--skip PHASE (phase name)] [--smoke] [--temporal-arm]
```

| Flag | What it does |
|---|---|
| `--fresh` | Archives the current `results/`, `models/` and the corresponding `_temporal` ones into `archives/run_<timestamp>/` together with `manifest.json`, and then deletes them so the run starts from scratch. Without it, `runall.py` does a **resume**: it reuses the previous results where the parameters and the data_fingerprint match. |
| `--only PHASE` | Runs **a single** phase (e.g. `--only tuning`). The others keep their previous status in `run_meta.json`. |
| `--skip PHASE (phase name)` | Skips one or more phases (e.g. `--skip seed_study specialist_study`). |
| `--smoke` | Produces synthetic datasets (900 rows/class, [`smoke_data.py`](smoke_data.py)) and redirects everything to `results_smoke/`, `models_smoke/`, `archives_smoke/`. Verifies that the whole pipeline runs, without needing real data. |
| `--temporal-arm` | After the main arm, repeats BoT-IoT and IoT-23 with a chronological split into `results_temporal/`and `models_temporal/`. |

Examples:

```bash
python src/runfull.py --fresh --temporal-arm
```

```bash
python src/runfull.py --only tuning
```

```bash
python src/runfull.py --skip seed_study specialist_study leakage_ablation
```

## `runall.py` (training only)

```bash
python src/runall.py [dataset] [--smoke] [--no-pooled]
```

* Without arguments: runs **all** the datasets found on disk.
* With dataset names: only those (e.g. `python src/runall.py bot_iot iot23`).
* **`--no-pooled`**: skips `combined` and `LODO`. It is done automatically in
  the temporal arm, because the pooled experiments make no sense when only 2 of the 4
  datasets run.

## `archiverun.py` (archiving)

```bash
python src/archiverun.py --list
```

```bash
python src/archiverun.py --fresh --label before_new_features
```

```bash
python src/archiverun.py --restore run_20260905_104454
```

`--restore` first keeps a copy of the current results.

## `status.py`

```bash
python status.py
```

It reads only `run_meta.json` and `progress.json`, has no dependencies, and is safe
to run while the pipeline is working.

## Individual phases

Each phase also runs on its own, with its own arguments:

```bash
python src/tuning.py bot_iot --trials 50 --models XGBoost LightGBM --task multiclass
```

```bash
python src/extendedstudy.py ciciot2023 --parts A C --cv-cap 30000
```

```bash
python src/seedstudy.py --seeds 42 7 13 34 99 --models RandomForest XGBoost
```
## `extra_studies.py` (standalone, not part of `runfull.py`)

Two extra studies that the main pipeline does not run. It is run by hand, **after** `run_all`, and writes everything to its own folder (`results_extra/` by default):

```bash
python src/extra_studies.py both
```

```bash
python src/extra_studies.py combined-tuning --trials 25 --tune-rows 30000
```

```bash
python src/extra_studies.py ciciot-merge --train-cap 250000 --slow-cap 60000
```

In short:

| Script | Positional arguments | Flags |
|---|---|---|
| `tuning.py` | `datasets` | `--task {binary,multiclass,both}` `--trials N` (25) `--models _` `--tune-rows N` (30000) |
| `extendedstudy.py` | `dataset` | `--parts A B C D` `--task` `--cv-cap N` (60000) `--cv-models _` |
| `seedstudy.py` | `datasets` | `--seeds _` (42 7 13 34) `--models _` `--tasks _` |
| `validationstats.py` | `datasets` | `--boot N` (500) |
| `specialiststudy.py` | `datasets` | `--parts A B C` |
| `costsensitive.py` | `datasets` | `--models _` `--costs _` (1 5 10) `--task` |
| `leakage_ablation.py` | `datasets` | `--models _` (RF, XGB, LGBM, CatBoost) |
| `operatingpoints.py` | `datasets` | — |
| `sanitycheck.py` | — | `--quiet` `--compare-archive RUN_ID` |

# Streamlit monitor

```bash
streamlit run src/trainingmonitor.py
```

It opens at **http://localhost:8501**.

**Pages:** `Run` (live progress per phase)  `Overview`,`Models`, `Tuning`,`Ablations`,`Generalisation`,`Files`.

**Sidebar:**

* **Run**: switch between the current run and any archived one (`archives/run_*`)
* **Refresh**: auto-refresh 3–30 seconds while the pipeline is running
* **Appearance**: Light / Dark theme

The dashboard is **read-only**: it reads `run_meta.json`, `progress.json`,
`all_results.json` and the PNGs of `results/figures/`. It can be opened while the
pipeline is running (`checkpoint()` writes atomically via a temporary file, so you
never read a half-written JSON), and also after it finishes, for the analysis.

> While a `temporal` phase is running, the monitor automatically reads `progress.json` from `results_temporal/`, while `run_meta.json` always stays in `results/`.

# Run times

Measured times from `results/run_meta.json` (run `pipeline_20260905_104457`,
16 GB RAM / RTX 4050 Laptop, all 4 datasets):

| Phase | Duration | % of the run |
|---|---:|---:|
| `run_all` | 1 h 24 m | 8.0% |
| `tuning` | 3 h 52 m | 22.1% |
| `extended_study` | 2 h 41 m | 15.3% |
| `deep_study` | 54 m 29 s | 5.2% |
| `anomaly_study` | 35 s | 0.1% |
| `adaptation_study` | 4 m 18 s | 0.4% |
| `deployment_bench` | 19 m 25 s | 1.8% |
| `extended_figures` | 18 m 28 s | 1.8% |
| `report_assets` | 4 s | <0.1% |
| `validation_stats` | 18 m 13 s | 1.7% |
| `domain_shift_study` | 6 m 12 s | 0.6% |
| `anomaly_sweep` | 22 s | <0.1% |
| `leakage_ablation` | 54 m 56 s | 5.2% |
| `operating_points` | 11 m 00 s | 1.0% |
| `cost_sensitive` | 1 h 01 m | 5.8% |
| `seed_study` | **3 h 11 m** | 18.2% |
| `specialist_study` | 17 m 17 s | 1.6% |
| `sanity_check` | 1 s | <0.1% |
| **Main arm subtotal** | **15 h 35 m** | 89% |
| `temporal_run_all` | 26 m 22 s | 2.5% |
| `temporal_report_assets` | 3 s | <0.1% |
| `temporal_validation_stats` | 8 m 03 s | 0.8% |
| `temporal_seed_study` | 1 h 21 m | 7.7% |
| `temporal_sanity_check` | 1 s | <0.1% |
| **Temporal arm subtotal** | **1 h 55 m** | 11% |
| **TOTAL** | **17 h 31 m** | 100% |

**Three phases consume 55% of the time:** `tuning` (Optuna, 25 trials × 3 folds per
model per dataset per task), `seed_study` (4 seeds × 15 models × 2 tasks ×
4 datasets) and `extended_study` (4 ablations × 2 tasks × 4 datasets).

The most expensive single model is CatBoost on CICIoT2023 multiclass (169 s, 250,000 rows), the cheapest, KNN (0.3 s it does not really train, it pays at inference).

# CUDA / GPU 

**The pipeline runs fully on CPU.** There is no step that *requires* a GPU.

### The only place that uses CUDA

**`TorchMLP`** ([`torchmlp.py`](torchmlp.py)) a PyTorch MLP (256→128→64, BatchNorm +
ReLU + Dropout 0.2, AdamW, early stopping with patience 6 on a 10% validation split),
wrapped in an sklearn-compatible `BaseEstimator`.

```python
def pick_device(self):
    if self.device:
        return self.device
    return "cuda" if torch.cuda.is_available() else "cpu"
```
## Three fallback levels

1. **There is no PyTorch at all**: `models_zoo.py` catches the `ImportError`,
   sets `HAS_TORCH = False` and `TorchMLP` simply does not enter the zoo.
   The pipeline runs with 18 instead of 19 models, without any error.
2. **There is PyTorch but no CUDA**: `torch.cuda.is_available()` returns `False`
   and training happens on CPU. Same results, longer time.
3. **There is CUDA but the VRAM fills up**: `fit()` explicitly catches `torch.cuda.OutOfMemoryError`,
   clears the cache and reruns the same training on CPU:

```python
try:
    self.fit_on(X, yi, ti, vi, d, k, device)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    self.fit_on(X, yi, ti, vi, d, k, "cpu")
```

Inference (`logits()`) is computed in **chunks of 8,192 rows** and the model
returns to the CPU after each call, so that it does not hold VRAM between phases.
The saved `.joblib` files always contain **CPU tensors** — they load on a machine without a GPU too.

## What does not use the GPU

**XGBoost, LightGBM and CatBoost run on CPU.** XGBoost uses
`tree_method="hist"` (histogram-based, CPU) with `n_jobs=-1` everywhere — **not** `device="cuda"`.
This was a deliberate choice: trees with ~250,000 rows × a few dozen features
are already fast on a multi-core CPU, and this way the results are reproducible on
any machine.

## Which phases touch `TorchMLP`

`run_all` · `extended_study` (ablations + learning curves) · `seed_study` ·
`cost_sensitive` · `leakage_ablation` · `specialist_study` · `extended_figures` (ROC/PR).
All the others are 100% CPU.

---

# Temporal split 

The question it answers: **if I train on old traffic, do I detect the new?**
The stratified split hides this, because flows of the same attack, from the same
seconds of capture, end up in both train and test.

## How it is implemented

All the logic is in `split()` of [`preprocessing.py`](preprocessing.py):

1. **Keeping the time.** Each loader calls `attach_timestamp()` and stores the
   timestamp in a **reserved `ts` column** (`TS_COL`). It accepts epoch seconds or parseable dates, and rejects "constant clocks" (`nunique <= 1`) that carry no information.
2. **`ts` is never a feature.** The first action of `split()` is
   `X = X.drop(columns=[TS_COL])`. **No model sees the time** (it serves only as a sort key).
3. **Deduplication** on the flow columns (excluding the windowed `w5_*` / `w10_*`), before the split.
4. **The chronological cut:**
```python
order = ts.fillna(ts.min() - 1).sort_values(kind="mergesort").index  # stable sort
cut = int(round(len(order) * (1 - TEST_SIZE)))                       # TEST_SIZE = 0.30
tr_idx, te_idx = order[:cut], order[cut:]
```

**The oldest 70% goes to training. The newest 30% to testing.** No stratification. The sort is `mergesort` (stable), so time
ties keep the file order and the split is deterministic.

## Two safety nets

| Condition | Reaction |
|---|---|
| There is no `ts`, or <10 valid values, or <2 unique | **Fallback to random** + warning |
| The cut leaves train with <2 classes (or test with 0) | **Fallback to random** + warning |

The fallback to random is always recorded, it never happens silently.

## The `split_audit.json`

Each split writes a record to `results/split_audit.json` with: the protocol requested and the one actually used, train/test sizes, classes in each, **classes missing
from train** (unlearnable), time ranges and `time_ranges_disjoint` (proof that
there is no overlap).

## Does the same apply to all datasets?

**No. A key finding of the protocol.** Only 2 of the 4 datasets
have a usable timestamp:

| Dataset | Time source | Temporal feasible? |
|---|---|---|
| **bot_iot** | `stime` / `ltime` (Argus) | ✅ Yes |
| **iot23** | `ts` (Zeek epoch) | ✅ Yes |
| **ton_iot** | `ts` present, but the dataset is **pre-balanced** (8 classes × 20,000) — the row sampling destroyed the temporal structure | ❌ Fallback to random |
| **ciciot2023** | no per-flow timestamp is distributed | ❌ Fallback to random |

That is why `--temporal-arm` runs **only** `bot_iot` and `iot23`
(`TEMPORAL_ARM_DATASETS` in [`runfull.py`](runfull.py)) and with `--no-pooled`: `combined`
and LODO would be built from a pool where half the datasets would have silently
fallen back to a random split, i.e. they would be **a hybrid that corresponds to no
protocol**.

## What it showed in practice

From `results_temporal/split_audit.json` of the last run:

```
bot_iot  temporal  train 280,290  test 120,124  disjoint=True  missing_from_train=['Theft']
iot23    temporal  train 111,726  test  47,883  disjoint=True  missing_from_train=[]
```

In BoT-IoT the **`Theft` class appears only in the test**: all the data theft
attacks were captured in the last 30% of the period. It is **impossible** to learn.
`split_meta()` marks it with **`evaluable = False`** and
`not_evaluable_reason = "classes absent from train: ['Theft']"`, and **every result
row carries this marking** so that it is excluded from the tables. In
`FIGURES_ORDER.md` the corresponding figures carry `[!]`.

This is exactly the argument of the temporal arm: the random split produced a
tidy macro-F1 for `Theft` and the chronological split revealed that
this number was a product of the protocol, not of detection ability.

> **Caution:** the `n_train` in the audit are recorded **before** the
> `MAX_TRAIN_ROWS = 250,000` cap is applied, which is why `bot_iot` shows 280,290.

## Manual activation (temporal only)

```bash
IOT_IDS_SPLIT=temporal IOT_IDS_RESULTS_DIR=results_temporal IOT_IDS_MODELS_DIR=models_temporal python src/runall.py bot_iot iot23 --no-pooled
```

In PowerShell:

```powershell
$env:IOT_IDS_SPLIT="temporal"; $env:IOT_IDS_RESULTS_DIR="results_temporal"; $env:IOT_IDS_MODELS_DIR="models_temporal"; python src/runall.py bot_iot iot23 --no-pooled
```

# Where the results are stored

Everything under **`src/`** (`PROJECT_DIR` in `config.py` is the folder of
`config.py` itself). All three roots can be overridden with environment variables.

```
src/
├── results/                          IOT_IDS_RESULTS_DIR (146 MB)
│   ├── all_results.json              ! 186 records: every (dataset, task, model) with all the metrics
│   ├── run_meta.json                 ! status + duration of each phase
│   ├── progress.json                 live progress (what the monitor reads)
│   ├── split_audit.json              ! which protocol each dataset actually got
│   ├── summary.csv                   summary table
│   │
│   ├── table_*.csv                   ! 33 tables:
│   │     table_overview_multiclass table_overview_binary table_per_class_<ds>
│   │     table_lodo table_tuning table_ablation table_learning_curves
│   │     table_confidence_intervals table_mcnemar table_baselines
│   │     table_cost_sensitive table_operating_points table_detection_at_fpr
│   │     table_leakage_ablation table_domain_shift table_seed_study
│   │     table_specialists table_cascade table_class_transfer
│   │     table_model_times table_selection_bias table_best_models  …
│   │
│   ├── <phase>_results.json            raw per study: tuning_results, extended_results,
│   │                                   deep_study, anomaly_results, adaptation_results,
│   │                                   domain_shift_study, leakage_ablation, operating_points,
│   │                                   cost_sensitive, seed_study, specialist_study,
│   │                                   validation_stats, deployment_bench (+ .csv)
│   ├── roc_pr_<dataset>.json           ROC/PR curve points (downsampled)
│   │
│   ├── figures/                      ! 312 PNG
│   │     confusion_<ds>__<task>__<model>.png
│   │     feature_importance_<ds>__<task>__<model>.png
│   │     class_distribution_<ds>.png model_comparison_<ds>.png
│   │     roc_*.png pr_*.png heatmap_*.png shap_*.png
│   │
│   ├── common_cache/<dataset>.pkl    the 11 common features, so that later
│   │                                 phases do not reload the datasets
│   ├── pipeline_<phase>.log          one log per phase (stdout + stderr together)
│   ├── pipeline_run.log              master log of all runs
│   └── <phase>.log                   the logger of each study
│
├── models/                           IOT_IDS_MODELS_DIR (1.3 GB)
│   ├── <dataset>__<task>__<model>.joblib     WHOLE pipeline (preprocessor + classifier)
│   └── best_models.json              ! the winners per task + per dataset (criterion: macro-F1)
│
├── results_temporal/  models_temporal/      same structure, temporal arm
├── results_matched/   models_matched/       matched-budget comparison (IOT_IDS_SLOW_CAP=250000)
│
├── results_extra/                    extra_studies.py: combined_tuning.json, ciciot_merge_study.json,
│
table_combined_*.csv, table_ciciot_merge*.csv, figures/, models/
└── archives/                         IOT_IDS_ARCHIVES_DIR
    └── run_<YYYYMMDD_HHMMSS>/
        ├── manifest.json             run_id, archiving time, number of files
        └── results/ models/ results_temporal/ models_temporal/
```

! = the files you need for writing the report.

## The two companion documents

* **[`FIGURES_ORDER.md`](FIGURES_ORDER.md) 402** figures (312 random + 90 temporal), marked `[!]` where `evaluable=False`.
* **[`TABLES_ORDER.md`](TABLES_ORDER.md) 38 tables**

Both follow the same ordering convention: datasets
`iot23, bot_iot, ton_iot, ciciot2023, combined`, models by family
(linear, bayes, instance, tree, bagging, boosting, neural), binary and multiclass.

# Environment variables

All are read in [`config.py`](config.py) and can be changed without editing code.

| Variable | Default | What it sets |
|---|---|---|
| `IOT_IDS_DATA_DIR` | *(auto-detect)* | Datasets folder (overrides the candidates) |
| `IOT_IDS_RESULTS_DIR` | `results` | Results folder|
| `IOT_IDS_MODELS_DIR` | `models` | Models folder |
| `IOT_IDS_ARCHIVES_DIR` | `archives` | Archive folder |
| **`IOT_IDS_SPLIT`** | `random` | `random` \| `temporal` |
| **`IOT_IDS_SLOW_CAP`** | `60000` | Row cap for the "slow" models (KNN, LinearSVM, MLP, DeepMLP, AdaBoost, TorchMLP). Set **`250000`** for the matched-budget comparison |
| `IOT_IDS_MIN_CLASS_LOAD` | `8000` | Minimum rows/class at load time (prevents rare attacks from disappearing in the capping) |
| `IOT_IDS_MIN_CLASS_SLOW` | `2000` | Same for the subsample of the slow models |
| `IOT_IDS_LOG_SCALE` | `1` | `sign(x)·log1p(\|x\|)` compression for heavy-tailed features `0` disables it |
| `IOT_IDS_FN_COST` | `1.0` | How many false alarms one missed attack is "worth" (`cost_sensitive`) |

Constants only inside the file: `MAX_ROWS_PER_DATASET=400000`, `MAX_TRAIN_ROWS=250000`,
`TEST_SIZE=0.30`, `RANDOM_STATE=42`, `MIN_CLASS_ROWS=60`.

# Resume, checkpointing & archiving

The pipeline is designed to **survive interruptions**

**Checkpoint after every model.** `checkpoint()` writes to `all_results.tmp` and
then does an atomic `replace()`. There is never a half-written `all_results.json`.

**Resume with fingerprint.** Before training, `train_all_models()` checks:

```python
prev is not None
  and dataset != "combined"
  and (MODELS_DIR / f"{ta}.joblib").exists()
  and prev["params"] == spec.params
  and prev["data_fingerprint"] == fingerprint
```

`data_fingerprint` is an MD5 of: number of training rows + sorted column
names + class distribution + the value of `SLOW_MODEL_TRAIN_CAP`. **If
anything changes in the data or the hyperparameters, the model is retrained.** If not,
it is reused and `resume <tag> previous results kept` is printed.

`combined` is **always excluded** from resume, because it depends on all the datasets together.

**Phase-level resume.** `runfull.py` reads the previous `run_meta.json` and
keeps the status of the phases not selected this time, so an `--only tuning` does not erase the history of the other 17 phases. `seed_study` additionally keeps its own `seed_study_partial.csv`.

**When to use `--fresh`:** after a change to the loaders, the preprocessing,
`model_zoo` or the constants of `config.py`. `--fresh` archives first and
deletes afterwards **no previous run is ever lost**.
# File map

## Core

| File | Role |
|---|---|
| [`config.py`](config.py) | Paths, limits, constants, 13-class taxonomy, 11 common features |
| [`loaders.py`](loaders.py) | The 4 loaders + generic loader, label unification· windowed features, `attach_timestamp`, `localize` |
| [`preprocessing.py`](preprocessing.py) | `build_xy`, `split` (random/temporal), `make_preprocessor`, `split_meta`, split audit |
| [`models_zoo.py`](models_zoo.py) | The `ModelSpec`s (11 basic + 4 optional), `LabelEncodedClassifier`, `build_with_class_weight` |
| [`torchmlp.py`](torchmlp.py) | PyTorch MLP with sklearn API, early stopping, CUDA, CPU fallback |
| [`evaluation.py`](evaluation.py) | `evaluate_model`: 20+ metrics, confusion matrix, ROC/PR, feature importance |
| [`progress.py`](progress.py) | The `progress.json` that feeds the monitor |

## Orchestration

| File | Role |
|---|---|
| [`runfull.py`](runfull.py) | The orchestrator of the 18 (+5) phases |
| [`runall.py`](runall.py) | The main training: per-dataset, combined, LODO |
| [`archiverun.py`](archiverun.py) | archive / list / restore / clear |
| [`sanitycheck.py`](sanitycheck.py) | Integrity check (exit 1 if it finds problems) |
| [`smoke_data.py`](smoke_data.py) | Synthetic datasets of 900 rows/class for `--smoke` |

## Studies

[`tuning.py`](tuning.py) [`extendedstudy.py`](extendedstudy.py) [`deepstudy.py`](deepstudy.py) [`anomalystudy.py`](anomalystudy.py) [`anomalysweep.py`](anomalysweep.py) [`adaptionstudy.py`](adaptionstudy.py) [`deploymentbench.py`](deploymentbench.py) [`domainshift_study.py`](domainshift_study.py) [`leakage_ablation.py`](leakage_ablation.py) [`operatingpoints.py`](operatingpoints.py) [`costsensitive.py`](costsensitive.py) [`seedstudy.py`](seedstudy.py) [`specialiststudy.py`](specialiststudy.py) [`validationstats.py`](validationstats.py)[`extra_studies.py`](extra_studies.py)

## Report material generation

[`reportassets.py`](reportassets.py) (CSV tables) [`make_extended_figures.py`](make_extended_figures.py)
(ROC/PR, heatmaps, SHAP) [`trainingmonitor.py`](trainingmonitor.py) (Streamlit dashboard)

# Reproducibility

**Fixed seed everywhere.** `RANDOM_STATE = 42` is passed to every estimator, every `train_test_split`,
every `sample()`, to `torch.manual_seed` and to `np.random.seed`. `seed_study`
explicitly measures how much the results move with seeds 42 / 7 / 13 / 34.

**No label leakage.** The columns `category` / `subcategory` / `attack` /
`detailed-label` are dropped right after `label` is produced. IP addresses,
`flow_id`s, `uid`s and timestamps are removed (`GENERIC_DROP`), the timestamp
survives only in `ts`, which `split()` drops before the model sees anything.

**The preprocessor is part of the pipeline.** `Pipeline([("pre", ...), ("clf", ...)])`
imputation, log-compression and scaling are learned **only** from the training fold, and
a new preprocessor is built per model. The saved `.joblib` contains
the **whole** pipeline, so deployment does not need to rebuild anything.

**Deduplication before the split**, on the flow columns (identical flows cannot be in train and test at the same time).

**Convergence check.** Every record carries `n_iter` and `converged`, so that it is visible
whether an iterative algorithm stopped because it converged or because the budget ran out.

**Honesty in the metrics.** Every result row carries `split_mode`,
`evaluable`, `not_evaluable_reason`, `classes_missing_from_train`, `classes_undertrained`,
`fully_learnable`, `train_min_class_rows`, `capped` and `train_cap`.

**`leakage_ablation`** quantifies how much of the performance was due to port numbers,
to BoT-IoT's per-IP aggregates and to windowed context features (information that does not generalise to another network).

# Troubleshooting
**`No datasets found. Place them under the dataset dir (see config.py)`**
`dataset_dir()` found nothing. Check with:

```bash
python -c "import sys; sys.path.insert(0,'src'); from loaders import available_datasets; from config import dataset_dir; print(dataset_dir()); print(available_datasets())"
```

Three possible causes, in this order of frequency:

1. **The `datasets/` folder is not in the right place** (next to `src/`
   or inside `src/`). Otherwise set `IOT_IDS_DATA_DIR`.
2. **The folder was found but the list is empty**: The files do not have a recognisable name. Rename them to `bot_iot.zip`, `ciciot2023.zip`, `iot_23.tar.gz`, `ton_iot.csv`.

**BoT-IoT loads 0 rows**
The ZIP does not contain the `Full5pc` CSVs. The loader deliberately ignores the `10-best features` you need `5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`.

**The monitor shows "stalled"**
`sanitycheck.py` treats **70 minutes** without an update of `progress.json` as a stall.
The `tuning` and `seed_study` phases can legitimately stay silent for a long time on a
heavy model. Confirm with `python status.py` or with the latest `pipeline_<phase>.log`.

**`PermissionError` on `all_results.json` (Windows)**
Streamlit keeps the file open. `checkpoint()` already makes 5 attempts with
a 0.3 s pause. If it persists, temporarily close the dashboard.

**A model is missing from the results**
Check `pipeline_run_all.log` for `Exception: skipping ...` or `MemoryError: skipping ...`.
Every model failure is logged and the pipeline moves on.

**`TorchMLP` does not appear anywhere**
PyTorch is not installed. Check with
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

**`OSError: not enough free space ... to copy`**
`localize()` tries to copy a dataset from another drive to
`datasets/local_cache/` and needs file size +20%.

**I want to confirm everything works before committing 20 hours**

```bash
python src/runfull.py --smoke
```

It runs the whole pipeline on synthetic data, in less time, without touching `results/` and `models/`.
