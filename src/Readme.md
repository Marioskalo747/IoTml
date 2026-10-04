# Ανίχνευση εισβολών σε δίκτυα IoT — Πειραματικό pipeline

Συγκριτική μελέτη μοντέλων μηχανικής μάθησης για ανίχνευση εισβολών σε δίκτυα IoT, σε **4 δημόσια datasets** (BoT-IoT, ToN-IoT, CICIoT2023, IoT-23), με **δύο
πρωτόκολλα διαχωρισμού** (τυχαίο / χρονολογικό), **15 μοντέλα** (19 διαμορφώσεις ανά dataset, με τις 4 βελτιστοποιημένες εκδοχές), δύο εργασίες
(δυαδική και πολυκλασική ανά κατηγορία επίθεσης) και **18**
πειραματικές φάσεις που παράγουν 186 εγγραφές αποτελεσμάτων, 33 πίνακες CSV και 402 γραφήματα.

Μια πλήρης εκτέλεση διαρκεί περίπου **20 ώρες** σε φορητό με 16 GB RAM και CUDA.
Η τελευταία καταγεγραμμένη εκτέλεση (`results/run_meta.json`) ολοκληρώθηκε σε
**17 ώρες 31 λεπτά** (15 ώ 35 λ ο κύριος βραχίονας + 1 ώ 55 λ ο χρονολογικός).

## Πίνακας περιεχομένων

1. [Ερευνητικά ερωτήματα](#ερευνητικά-ερωτήματα)
2. [Γρήγορη εκκίνηση](#γρήγορη-εκκίνηση)
3. [Απαιτήσεις συστήματος](#απαιτήσεις-συστήματος)
4. [Δομή φακέλων](#δομή-φακέλων)
5. [Datasets](#datasets)
6. [Μέθοδος](#μέθοδος)
7. [Πώς εκτελείται το pipeline](#πώς-εκτελείται-το-pipeline)
8. [Εντολές & παράμετροι](#εντολές--παράμετροι)
9. [Streamlit monitor](#streamlit-monitor)
10. [Χρόνοι εκτέλεσης](#χρόνοι-εκτέλεσης)
11. [CUDA / GPU](#cuda--gpu)
12. [Χρονολογικός διαχωρισμός](#χρονολογικός-διαχωρισμός)
13. [Πού αποθηκεύονται τα αποτελέσματα](#πού-αποθηκεύονται-τα-αποτελέσματα)
14. [Μεταβλητές περιβάλλοντος](#μεταβλητές-περιβάλλοντος)
15. [Συνέχιση, checkpointing & αρχειοθέτηση](#συνέχιση-checkpointing--αρχειοθέτηση)
16. [Χάρτης αρχείων](#χάρτης-αρχείων)
17. [Αναπαραγωγιμότητα](#αναπαραγωγιμότητα)
18. [Γνωστοί περιορισμοί](#γνωστοί-περιορισμοί)
19. [Αντιμετώπιση προβλημάτων](#αντιμετώπιση-προβλημάτων)


# Ερευνητικά ερωτήματα

Η αρίθμηση είναι ίδια με της πτυχιακής (ΕΕ1–ΕΕ8).

| # | Ερώτημα | Script(s) |
|---|---|---|
| RQ1 | Πόσο καλά ανιχνεύουν και ταξινομούν τις επιθέσεις 15 ταξινομητές σε 4 IoT datasets και στην ένωσή τους, με τον στρωματοποιημένο τυχαίο διαχωρισμό της βιβλιογραφίας; | `runall.py`, `tuning.py` |
| RQ2 | Είναι πραγματικές οι διαφορές μεταξύ των καλύτερων μοντέλων, και αλλάζει ο νικητής όταν το μοντέλο επιλέγεται με διασταυρούμενη επικύρωση αντί για το test split; | `validationstats.py`, `seedstudy.py`, `extendedstudy.py` (μέρος D), `validated_selection.py` (`cv`) |
| RQ3 | Πόση από την απόδοση προέρχεται από χαρακτηριστικά που αντανακλούν το εργαστήριο (θύρες, συγκεντρωτικά ανά IP, window context, η ίδια η ταυτότητα του dataset); | `leakage_ablation.py`, `domainshift_study.py` |
| RQ4 | Τι απομένει από την απόδοση σε δίκτυο που το μοντέλο δεν έχει δει, και μπορεί η ανίχνευση ανωμαλιών να βρει άγνωστες επιθέσεις; | LODO στο `runall.py`, `domainshift_study.py`, `specialiststudy.py` (μέρος C), `anomalystudy.py`, `anomalysweep.py` |
| RQ5 | Τι συμβαίνει όταν το μοντέλο εκπαιδεύεται στο παρελθόν και αξιολογείται στο μέλλον; | χρονολογικός βραχίονας (`runfull.py --temporal-arm`) |
| RQ6 | Βοηθούν οι συνήθεις τεχνικές βελτίωσης (βελτιστοποίηση υπερπαραμέτρων, στρατηγικές ανισορροπίας, εκπαίδευση με κόστος, επιλογή χαρακτηριστικών, εξειδικευμένοι ανιχνευτές, κλιμακωτά σχήματα); | `tuning.py`, `extra_studies.py`, `confirm_tuning_seeds.py`, `extendedstudy.py` (A, B), `deepstudy.py`, `costsensitive.py`, `specialiststudy.py` (A, B) |
| RQ7 | Πόσα δεδομένα από το δίκτυο-στόχο χρειάζονται για να ανακτηθεί η απόδοση που χάνεται μεταξύ δικτύων; | `adaptionstudy.py` |
| RQ8 | Τι σημαίνουν οι μετρικές σε λειτουργικούς όρους: ακρίβεια των συναγερμών σε ρεαλιστική επικράτηση επιθέσεων, κατώφλια επιλεγμένα σε δεδομένα επικύρωσης, ταχύτητα και μέγεθος μοντέλου; | `operatingpoints.py`, `validated_selection.py` (`thresholds`), `deploymentbench.py` |

Η πλήρης συζήτηση κάθε ερωτήματος και της απάντησής του βρίσκεται στην πτυχιακή. Το αρχείο αυτό περιγράφει πώς εκτελούνται και αναπαράγονται τα πειράματα.


# Γρήγορη εκκίνηση

**0.** Όλες οι εντολές
εκτελούνται **από τον γονικό φάκελο**. Λεπτομέρειες:
[Δομή φακέλων](#δομή-φακέλων).

```
project/
├── src/          <- the code
└── datasets/     <- the data
```

```bash
pip install -r src/requirements.txt
```

Το [`requirements.txt`](requirements.txt) καθορίζει τις **ακριβείς εκδόσεις** με τις οποίες παράχθηκαν όλα
τα αποτελέσματα. Για την έκδοση CUDA του PyTorch (προαιρετική· αν λείπει, όλα τρέχουν σε CPU):
`pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`.

**1. Δοκιμή χωρίς δεδομένα** (συνθετικά datasets, γράφει στο `results_smoke/`):

```bash
python src/runfull.py --smoke
```

**2. Πλήρης εκτέλεση** (χρειάζεται τα πραγματικά datasets, βλ. [Datasets](#datasets)):

```bash
python src/runfull.py --fresh --temporal-arm
```

**3. Παρακολούθηση σε άλλο τερματικό**:

```bash
streamlit run src/trainingmonitor.py
```

**4. Αυτόνομες μελέτες** (μετά το βήμα 2, με αυτή τη σειρά· βλ. [Εντολές & παράμετροι](#εντολές--παράμετροι)):

```bash
python src/extra_studies.py
```

```bash
python src/validated_selection.py
```

```bash
python src/confirm_tuning_seeds.py --seeds 42 7 13 34 101 202 303 404 505 606
```

# Απαιτήσεις συστήματος

## Λογισμικό

| | Έκδοση δοκιμής | Σημείωση |
|---|---|---|
| Python | **3.14.0** (Windows 11, 64-bit) | λειτουργεί και σε 3.11+ |
| numpy, pandas, scipy | 2.3.5 / 2.3.3 / 1.16.3 | |
| scikit-learn | 1.7.2 | ο πυρήνας όλων των pipelines |
| imbalanced-learn | 0.14.0 | SMOTE / over- / under-sampling |
| xgboost, lightgbm, catboost | 3.1.3 / 4.6.0 / 1.2.10 | **προαιρετικά**· αν λείπουν, το `model_zoo` απλώς δεν τα προσθέτει |
| torch | 2.9.1+cu128 | **προαιρετικό**· μόνο για το μοντέλο `TorchMLP` |
| optuna | 4.9.0 | φάση `tuning` (TPE sampler) |
| matplotlib, seaborn, plotly | 3.10.7 / 0.13.2 / 6.5.0 | γραφήματα PNG + διαδραστικά γραφήματα στο monitor |
| streamlit | 1.58.0 | dashboard |
| joblib | 1.5.2 | αποθήκευση μοντέλων (`compress=3`) |

Όλες οι εκδόσεις του πίνακα βρίσκονται στο [`requirements.txt`](requirements.txt).

Οι προαιρετικές βιβλιοθήκες ανιχνεύονται με `try/except ImportError` στο
`models_zoo.py`. Το pipeline τρέχει και χωρίς αυτές, απλώς με λιγότερα μοντέλα.

## Υλικό

| Πόρος | Ελάχιστο | Προτεινόμενο (μηχάνημα αναφοράς) |
|---|---|---|
| **RAM** | 8 GB (με μειωμένα όρια) | **16 GB** |
| CPU | 4 πυρήνες | 8+ πυρήνες (τα περισσότερα μοντέλα τρέχουν με `n_jobs=-1`) |
| **GPU/CUDA** | **προαιρετικό** | NVIDIA RTX 4050 Laptop (6 GB VRAM), CUDA 12.8 |
| Δίσκος | 5 GB ελεύθερα | 10 GB (μοντέλα 1,3 GB + αποτελέσματα 146 MB + ένα αρχείο ανά εκτέλεση) |

**Γιατί 16 GB RAM:** το όριο φόρτωσης είναι `MAX_ROWS_PER_DATASET = 400,000`
γραμμές ανά dataset και `MAX_TRAIN_ROWS = 250,000` γραμμές εκπαίδευσης μετά τον
διαχωρισμό ([`config.py`](config.py)). Το dataset `combined` φτάνει τα 1,08 εκατομμύρια γραμμές.
Η κορύφωση συμβαίνει στο `run_all` (φόρτωση + προεπεξεργασία + `RandomForest`/`ExtraTrees`
με 200 δέντρα) και στο `seed_study` (4 σπόροι × 15 μοντέλα).

Το `train_all_models()` πιάνει το `MemoryError` ανά μοντέλο: αν ένα μοντέλο δεν χωράει, παραλείπεται, καλείται το `gc.collect()` και το pipeline συνεχίζει αντί να καταρρεύσει.

**Αν έχετε λιγότερη RAM**, μειώστε τα όρια πριν από την εκτέλεση:

```bash
IOT_IDS_SLOW_CAP=30000 IOT_IDS_MIN_CLASS_LOAD=4000 python src/runfull.py
```

ή αλλάξτε απευθείας τα `MAX_ROWS_PER_DATASET` / `MAX_TRAIN_ROWS` στο [`config.py`](config.py).

# Δομή φακέλων

Το σύστημα στον γονικό φάκελο αποτελείται από τα `src/` και `datasets/` και τους υποφακέλους `results_*/` του src

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

## Ο φάκελος πρέπει να λέγεται `src`

Το [`runfull.py`](runfull.py) σχηματίζει τις διαδρομές των φάσεων ως
`<parent>/src/<script>.py` (`ROOT = Path(__file__).resolve().parents[1]`).

Οι **υπόλοιπες** εντολές (`runall.py`, `tuning.py`, όλες οι μελέτες) λειτουργούν με
οποιοδήποτε όνομα φακέλου, γιατί χρησιμοποιούν `Path(__file__).parent`. **Μόνο
το `runfull.py` έχει αυτή την απαίτηση** — και είναι η κύρια εντολή, οπότε κρατήστε το όνομα `src`.

## Τι χρειάζεται αλλαγή στα αρχεία

**Στην κανονική περίπτωση, τίποτα.** Ο μόνος λόγος να αγγίξετε το [`config.py`](config.py)
είναι η σταθερή διαδρομή δεδομένων του μηχανήματος ανάπτυξης:

```python
DATASET_DIR_CANDIDATES = [Path(r"path/to/datasets/in/files"), 
PROJECT_DIR.parent / "datasets",     # <parent>/datasets
PROJECT_DIR / "datasets"]            # src/datasets
```

Επιστρέφει τον πρώτο φάκελο που **υπάρχει**.
Μπορείτε να αντικαταστήσετε την πρώτη γραμμή με τη δική σας διαδρομή προς τα datasets.

Τρεις τρόποι να δηλώσετε πού βρίσκονται τα δεδομένα, από τον απλούστερο:

| # | Τρόπος | Πότε |
|---|---|---|
| 1 | Δημιουργία `datasets/` **δίπλα στο `src/`** | η προτεινόμενη λύση, χωρίς αλλαγή κώδικα |
| 2 | Δημιουργία `datasets/` **μέσα στο `src/`** | αν έχετε μόνο τον φάκελο `src` και δεν θέλετε τίποτα έξω από αυτόν |
| 3 | Ορισμός του `IOT_IDS_DATA_DIR` | αν τα δεδομένα είναι σε άλλο δίσκο ή σε κοινόχρηστο φάκελο |

```bash
IOT_IDS_DATA_DIR=/mnt/data/iot_datasets python src/runfull.py
```

```powershell
$env:IOT_IDS_DATA_DIR="D:\iot_datasets"; python src\runfull.py
```

Το `IOT_IDS_DATA_DIR` **υπερισχύει όλων** των υποψηφίων, άρα είναι και ο τρόπος να παρακάμψετε τη σταθερή διαδρομή χωρίς να αγγίξετε τον κώδικα.

## Έλεγχος πριν από την εκτέλεση

```bash
python -c "import sys; sys.path.insert(0,'src'); from config import dataset_dir, PROJECT_DIR, RESULTS_DIR; from loaders import available_datasets; print('code   :', PROJECT_DIR); print('data   :', dataset_dir()); print('results:', RESULTS_DIR); print('found  :', available_datasets())"
```

Σωστή έξοδος:

```
code   : ...\project\src
data   : ...\project\datasets
results: ...\project\src\results
found  : ['bot_iot', 'ton_iot', 'ciciot2023', 'iot23']
```

Αν το `found` είναι κενή λίστα, τα αρχεία δεν έχουν τα ονόματα που περιμένει ο κώδικας.


# Datasets

## Επίσημες πηγές και αρχεία που χρησιμοποιήθηκαν

| Dataset | Επίσημη σελίδα | Τι χρησιμοποιήθηκε | Όνομα στο `datasets/` | Τι διαβάζει ο loader | Δημοσίευση για αναφορά |
|---|---|---|---|---|---|
| **IoT-23** | <https://www.stratosphereips.org/datasets-iot23> (και Zenodo, [doi:10.5281/zenodo.4743746](https://doi.org/10.5281/zenodo.4743746)) | η ελαφριά έκδοση του αρχείου: μόνο τα Zeek logs `conn.log.labeled`, χωρίς τα pcaps | `iot_23.tar.gz` (9,37 GB) | όλα τα `opt/Malware-Project/BigDataset/IoTScenarios/CTU-IoT-Malware-Capture-*/bro/conn.log.labeled`, έως 60.000 ροές ανά σενάριο | S. Garcia, A. Parmisano, M. J. Erquiaga, *IoT-23: A labeled dataset with malicious and benign IoT network traffic*, Stratosphere Lab, 2020 |
| **BoT-IoT** | <https://research.unsw.edu.au/projects/bot-iot-dataset> | ο φάκελος `5%` της διανομής (επίσημο υποσύνολο 5%), συμπιεσμένος σε zip | `bot_iot.zip` (1,99 GB) | **μόνο** τα `5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`· τα `10-best features` αγνοούνται | N. Koroniotis et al., *Future Generation Computer Systems*, vol. 100, 2019 |
| **ToN-IoT** | <https://research.unsw.edu.au/projects/toniot-datasets> | το CSV κίνησης δικτύου των συνόλων train/test, `Train_Test_Network.csv` (211.043 ροές, στήλες `label` και `type`) | `ton_iot.csv` (29,9 MB) | όλες οι γραμμές· ετικέτα από τη στήλη `type` | N. Moustafa, *Sustainable Cities and Society*, vol. 72, 2021· A. Alsaedi et al., *IEEE Access*, vol. 8, 2020 |
| **CICIoT2023** | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | τα CSV χαρακτηριστικών ανά καταγραφή (`<Attack>/<name>.pcap.csv`, 34 φάκελοι: 33 επιθέσεις + Benign), συμπιεσμένα σε zip με ρίζα `CSV/` | `ciciot2023.zip` (1,43 GB) | όλα τα `CSV/<Attack>/*.pcap.csv`, με προϋπολογισμό γραμμών **ανά κλάση** (`ciciot_budget`)· ετικέτα από τη στήλη `label` αν υπάρχει, αλλιώς από το όνομα του φακέλου | E. C. P. Neto et al., *Sensors*, vol. 23, no. 13, 2023 |

Οι πλήρεις βιβλιογραφικές αναφορές και η περιγραφή κάθε συνόλου βρίσκονται στην τεκμηρίωση της πτυχιακής.
Η χρήση κάθε συνόλου απαιτεί αναφορά στις δημοσιεύσεις που ζητά η πηγή του, και οι όροι
χρήσης και αναδιανομής πρέπει να ελέγχονται στην επίσημη σελίδα.

## Η διάταξη που λειτουργεί

Κατεβάστε τα 4 datasets από τις επίσημες σελίδες τους, **μετονομάστε** τα αρχεία και
αφήστε τα **ως έχουν, συμπιεσμένα**, σε έναν φάκελο `datasets/`:

```
datasets/
├── bot_iot.zip        1.99 GB   BoT-IoT        — UNSW Canberra
├── ciciot2023.zip     1.43 GB   CICIoT2023     — Canadian Institute for Cybersecurity (UNB)
├── iot_23.tar.gz      9.37 GB   IoT-23         — Stratosphere Laboratory (CTU)
└── ton_iot.csv       29.9 MB    ToN-IoT        — UNSW Canberra
                     ─────────
                      12.8 GB in total
```

Αυτή είναι ακριβώς η δομή του μηχανήματος ανάπτυξης και η **απλούστερη που
ικανοποιεί όλα τα μοτίβα**. Δεν χρειάζεται αποσυμπίεση· οι loaders διαβάζουν με
streaming απευθείας μέσα από το zip / tar.gz, γι' αυτό δεν χρειάζεστε 50 GB ελεύθερου χώρου για αποσυμπίεση.

## Τι δέχεται κάθε loader

Η αναζήτηση είναι **αναδρομική** (`rglob`), οπότε τα αρχεία μπορούν να βρίσκονται και σε
υποφακέλους του `datasets/`, αρκεί το **όνομα** να ταιριάζει με ένα από τα μοτίβα
(`BUILTIN_PROBES` στο [`loaders.py`](loaders.py)):

| Dataset | Αναγνωριζόμενα ονόματα | Τι πρέπει να περιέχει |
|---|---|---|
| **bot_iot** | `bot_iot*.zip` `5%*.zip` `*5pc*.zip` `*bot*iot*.zip` `*Full5pc*.csv` | Τα CSV του **υποσυνόλου 5%**: `5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`. Ο loader κρατά **μόνο** τα μέλη που περιέχουν `Full5pc`· τα `10-best features` αγνοούνται |
| **ton_iot** | `ton_iot*.csv` `Train_Test_Network*.csv` `train_test_network*.csv` | Το μοναδικό CSV `Train_Test_Network.csv` (στήλη `type` για την ετικέτα). Είναι ήδη ισορροπημένο: 8 κλάσεις × 20.000 |
| **ciciot2023** | `ciciot2023*.zip` `CSV.zip` `*CICIoT*.zip` `*ciciot*.zip` `*.pcap.csv` | Τα CSV ανά επίθεση: `CSV/<Attack_Name>/<Attack_Name>.pcap.csv`. Η ετικέτα διαβάζεται από τη στήλη `label` αν υπάρχει, αλλιώς από το όνομα του φακέλου |
| **iot23** | `iot_23*.tar.gz` `*iot*23*.tar.gz` `conn.log.labeled` `*.log.labeled` | Τα Zeek logs `.../IoTScenarios/CTU-IoT-Malware-Capture-<N>-<M>/bro/conn.log.labeled` |

**Δύο εναλλακτικές αν έχετε ήδη αποσυμπιέσει:**

* CICIoT2023: αφήστε τα `*.pcap.csv` μέσα στους φακέλους ανά επίθεση και ο loader τα βρίσκει.
* IoT-23: αφήστε τα `conn.log.labeled` στη δομή φακέλων τους.

**Δεν χρειάζεστε και τα 4.** Το `available_datasets()` επιστρέφει όσα βρήκε και το pipeline
τρέχει με αυτά. Με λιγότερα από 2 datasets, όμως, τα `combined` και `LODO` παραλείπονται
αυτόματα (`not enough datasets for combined model`).

## Datasets σε άλλο δίσκο

Αν το `IOT_IDS_DATA_DIR` δείχνει σε **διαφορετικό γράμμα δίσκου** από τον κώδικα,
το `localize()` αντιγράφει πρώτα το αρχείο στο `<parent>/datasets/local_cache/` — με
έλεγχο ελεύθερου χώρου (περιθώριο +20%) και 3 προσπάθειες. Το streaming των
συμπιεσμένων αρχείων από εξωτερικό ή δικτυακό δίσκο είναι πολύ πιο αργό από την αντιγραφή.

> Προσοχή στον χώρο: η cache δημιουργείται **πάντα** δίπλα στον κώδικα, όχι στον δίσκο των δεδομένων. Για το `iot_23.tar.gz` πρέπει να υπάρχουν εκεί 11 GB ελεύθερα.

## Προσθήκη δικού σας dataset

Σε αντίθεση με τα ενσωματωμένα, ένα δικό σας dataset **πρέπει** να βρίσκεται σε δικό του υποφάκελο
του `datasets/`, και το όνομα του φακέλου γίνεται το όνομα του dataset:

```
datasets/
└── myset/                  <- the dataset name
    ├── part1.csv
    └── part2.csv
```

Για να το ενεργοποιήσετε, δημιουργήστε ένα αρχείο **`<parent>/datasets_config.json`** **δίπλα στο `src/`, όχι μέσα του**:

```json
{ "custom": { "myset": { "enabled": true, "label_col": "attack_type", "glob": "*.csv" } } }
```

Το `label_col` είναι προαιρετικό (αν λείπει, το `detect_label_col()` αναζητά τα συνήθη
ονόματα `label`, `attack`, `attack_type`, `attack_cat`, `category`, `type`, `class`, …).
Το `generic_loader()` αναλαμβάνει την ενοποίηση της ταξινομίας (`unify_label`) και την εξαγωγή
των 11 κοινών χαρακτηριστικών. Η ενεργοποίηση είναι **ρητή (opt-in)**:
το `discover_candidates()` βλέπει τον φάκελο, αλλά χωρίς `"enabled": true` τίποτα δεν μπαίνει στα
πειράματα κατά λάθος.

## Προεπεξεργασία

Όλη η προεπεξεργασία γίνεται από τον κώδικα, **κάθε φορά από τα ακατέργαστα αρχεία**. Δεν χρειάζεται κανένα ενδιάμεσο «καθαρισμένο» αρχείο που να φτιάχνεται με το χέρι. Τα βήματα, με τη σειρά:

**1. Loader ανά dataset** ([`loaders.py`](loaders.py)):

| Dataset | Ετικέτα | Τι αφαιρείται | Τι αλλάζει |
|---|---|---|---|
| BoT-IoT | `category` `BOTIOT_LABEL_MAP` (normal, Benign, DDoS, DoS, reconnaissance, Recon, theft, Theft) | `pkSeqID`, `stime`, `ltime`, `seq`, `saddr`, `daddr` και οι στήλες ετικετών `attack`, `category`, `subcategory` (`BOTIOT_DROP`) | 12 χαρακτηριστικά πλαισίου (παράθυρα 5 και 10 s) από IP και χρόνο, πριν αφαιρεθούν (δεκαεξαδικές θύρες σε δεκαδικές (`parse_port`)) |
| ToN-IoT | `type` `TONIOT_LABEL_MAP` (10 κλάσεις)· η αρχική δυαδική στήλη `label` αντικαθίσταται από την ενοποιημένη ετικέτα | IPs, `ts` και πεδία κειμένου υψηλής πληθικότητας: `dns_query`, `http_uri`, `http_user_agent`, `ssl_subject`, `ssl_issuer`, `weird_*` κ.λπ. (`TONIOT_DROP`) | η παύλα - του Zeek γίνεται NaN |
| CICIoT2023 | όνομα φακέλου (ή στήλη `label`) `ciciot_map_label` (10 κλάσεις) | δεν έχει IPs ή θύρες | ανάγνωση σε τμήματα των 250.000 γραμμών με προϋπολογισμό ανά κλάση |
| IoT-23 | `label` + `detailed-label` `iot23_map_label` (4 κλάσεις) | IPs, `uid`, `ts`, `history`, `local_orig`, `local_resp`, `tunnel_parents`, `detailed-label` | 12 χαρακτηριστικά πλαισίου σε διαδοχικά μπλοκ γραμμών (`_windowed_blocks`) |

Και στα τέσσερα: κλάσεις με λιγότερες από 60 ροές συγχωνεύονται στην `Other` (`merge_rare_classes`),
και κάθε σύνολο περιορίζεται, στρωματοποιημένα, στις 400.000 ροές με ελάχιστο 8.000 ανά κλάση
(`stratified_cap`). Η χρονοσφραγίδα διατηρείται μόνο στη στήλη `ts`, για τον χρονολογικό διαχωρισμό.

**2. Κοινό σχήμα 11 χαρακτηριστικών** (οι συναρτήσεις `*_common` στο [`loaders.py`](loaders.py)):
`duration`, `total_pkts`, `total_bytes`, `pkt_rate`, `byte_rate`, `avg_pkt_size`,
`proto_tcp/udp/icmp`, `dst_port`, `src_port`. Ρυθμός 0 για ροές μηδενικής διάρκειας. Στο CICIoT2023
οι θύρες είναι 0 και ο ρυθμός πακέτων είναι το εγγενές πεδίο `rate`. Αποθηκεύεται στο
`results/common_cache/<dataset>.pkl` και τροφοδοτεί την ένωση (`combined`), το LODO και τις
μελέτες μεταφοράς, προσαρμογής και ανωμαλιών.

**3. Διαχωρισμός** (`split()` στο [`preprocessing.py`](preprocessing.py)):

1. αφαίρεση ακριβών διπλοτύπων (στήλες ροής + ετικέτα) **πριν** από τον διαχωρισμό.
2. στρωματοποιημένος διαχωρισμός 70/30 με σπόρο 42 (ή χρονολογικός, με `IOT_IDS_SPLIT=temporal`).
3. όριο εκπαίδευσης 250.000 γραμμών, αναλογικά ανά κλάση με ελάχιστο 1.000, χωρίς περιορισμό του test.
4. κρατιούνται οι 30 συχνότερες τιμές κάθε κατηγορικής στήλης, οι υπόλοιπες γίνονται `RARE`.
5. αφαίρεση στηλών με μία μόνο τιμή στο train.

Τα αργά μοντέλα (k-NN, LinearSVM, AdaBoost, MLP, DeepMLP, TorchMLP) εκπαιδεύονται σε στρωματοποιημένο
υποδείγμα 60.000 γραμμών, με ελάχιστο 2.000 ανά κλάση.

**4. Μετασχηματιστής ανά μοντέλο** (`make_preprocessor()`, μέσα σε `Pipeline`, `fit` μόνο στο train):
αριθμητικά -> median imputation + δείκτες ελλείπουσας τιμής -> `sign(x)·log1p(|x|)` -> `StandardScaler` και
κατηγορικά -> most-frequent + one-hot. Κάθε στήλη θύρας δίνει επιπλέον 19 δείκτες (4 εύρη και 15
υπηρεσίες: 21, 22, 23, 25, 53, 80, 123, 443, 445, 502, 1883, 3389, 5683, 8080, 8883).

## Επεξεργασμένα υποσύνολα και όροι χρήσης

Το αποθετήριο **δεν περιέχει δεδομένα**: τα `datasets/`, `results/` και όλα τα `*.pkl` βρίσκονται στο Drive. Το pipeline παράγει ένα επεξεργασμένο υποσύνολο ανά dataset,
το `results/common_cache/<dataset>.pkl`: τα 11 κοινά χαρακτηριστικά μαζί με τις στήλες `label`, `binary`
και `source`, μετά τη δειγματοληψία των loaders. Τα υποσύνολα αυτά **δεν
δημοσιεύονται**, για δύο λόγους:

1. **Όροι χρήσης.** Είναι παράγωγα των αρχικών συνόλων, και κάθε πηγή ορίζει τους δικούς της όρους
   χρήσης και αναδιανομής.
2. **Δεν αρκούν για αναπαραγωγή.** Περιέχουν μόνο το κοινό σχήμα. Τα μοντέλα ανά dataset χρησιμοποιούν
   τα εγγενή χαρακτηριστικά, που υπάρχουν μόνο στα ακατέργαστα αρχεία. Από την cache
   θα μπορούσαν να ξανατρέξουν μόνο οι μελέτες του κοινού σχήματος (`domain_shift_study`, `adaptation_study`,
   `anomaly_study`, `anomaly_sweep`).

Η πλήρης αναπαραγωγή γίνεται από τα επίσημα αρχεία με τις εντολές της
[§ Γρήγορη εκκίνηση](#γρήγορη-εκκίνηση). Η αλυσίδα είναι ντετερμινιστική: τα ακατέργαστα αρχεία ξαναφορτώνονται και αναπαράγονται ακριβώς οι ίδιοι διαχωρισμοί και οι ίδιες μετρικές.

# Μέθοδος

## Κοινή ταξινομία ετικετών

Οι ετικέτες κάθε dataset αντιστοιχίζονται σε μια **κοινή ταξινομία 13 κλάσεων**
(`config.py`): `Benign`, `DDoS`, `DoS`, `Mirai`, `Botnet`, `Recon`, `BruteForce`, `Web`,
`Spoofing_MITM`, `Injection`, `Ransomware`, `Theft`, `Other`. Κλάσεις επίθεσης με λιγότερες από 60 ροές
συγχωνεύονται στην `Other`· αν η ίδια η `Other` μείνει κάτω από 60 ροές, αφαιρείται. Στα τελικά δεδομένα
καμία κλάση δεν καταλήγει στην `Other`, οπότε χρησιμοποιούνται 12 κλάσεις.

| Κοινή κλάση | BoT-IoT | ToN-IoT | CICIoT2023 | IoT-23 |
|---|---|---|---|---|
| Benign | Normal | normal | Benign | benign |
| DDoS | DDoS | ddos | DDoS-\* | DDoS |
| DoS | DoS | dos | DoS-\* | — |
| Recon | Reconnaissance | scanning | Recon-\*, VulnerabilityScan | PartOfAHorizontalPortScan |
| Theft | Theft | — | — | — |
| BruteForce | — | password | DictionaryBruteForce | — |
| Web | — | xss | XSS, Uploading, BrowserHijacking | — |
| Injection | — | injection | SqlInjection, CommandInjection | — |
| Botnet | — | backdoor | Backdoor_Malware | C&C, Okiru, Torii, FileDownload, … |
| Ransomware | — | ransomware | — | — |
| Spoofing_MITM | — | mitm | MITM-ArpSpoofing, DNS_Spoofing | — |
| Mirai | — | — | Mirai-\* | — |

Παράδειγμα: στο ToN-IoT το `scanning` γίνεται `Recon` και το `password` γίνεται `BruteForce`. Η αντιστοίχιση δεν είναι
πάντα σημασιολογικά ισοδύναμη (π.χ. η `Botnet` είναι backdoor στο ToN-IoT και κίνηση C&C στο IoT-23)· είναι ο
συμβιβασμός που κάνει δυνατή τη σύγκριση και τη συνένωση.

## Μεγέθη των datasets μετά τον διαχωρισμό

Τυχαίος διαχωρισμός, πολυκλασική εργασία (`results/all_results.json`). Το «Train» είναι μετά το όριο των 250.000 γραμμών,
τα «Χαρακτηριστικά» είναι το πλήθος των εισόδων του μοντέλου μετά την προεπεξεργασία.

| Dataset | Train (πριν το όριο) | Train | Test | Χαρακτηριστικά | Κλάσεις στο test |
|---|---:|---:|---:|---:|---|
| BoT-IoT | 280.289 | 250.040 | 120.125 | 105 | Benign, DDoS, DoS, Recon, Theft |
| ToN-IoT | 124.224 | 124.224 | 53.240 | 94 | Benign + 9 επιθέσεις |
| CICIoT2023 | 263.118 | 249.995 | 112.765 | 42 | Benign + 9 επιθέσεις |
| IoT-23 | 111.726 | 111.726 | 47.883 | 86 | Benign, Botnet, DDoS, Recon |
| combined | 754.324 | 250.686 | 323.283 | 49 (από τα 11 κοινά χαρακτηριστικά) | 12 κλάσεις |

## Μοντέλα και βελτιστοποίηση

15 μοντέλα από 7 οικογένειες ([`models_zoo.py`](models_zoo.py)). Τα τελευταία τέσσερα είναι προαιρετικά: χωρίς
`xgboost`, `lightgbm`, `catboost` ή `torch` το αντίστοιχο μοντέλο απλώς δεν προστίθεται.

| Μοντέλο | Οικογένεια | Κύριες ρυθμίσεις | Αργό (60.000 γραμμές) |
|---|---|---|---|
| LogisticRegression | γραμμικό | C=1.0, max_iter=2000 | |
| GaussianNB | Bayes | προεπιλογές | |
| DecisionTree | δέντρο | max_depth=25, min_samples_leaf=3 | |
| RandomForest | bagging | 200 δέντρα, min_samples_leaf=2 | |
| ExtraTrees | bagging | 200 δέντρα, min_samples_leaf=2 | |
| HistGradientBoosting | boosting | max_iter=200, learning_rate=0.1 | |
| AdaBoost | boosting | 100 δέντρα βάθους 3 | ✓ |
| KNN | βάσει παραδειγμάτων | k=7 | ✓ |
| LinearSVM | SVM | C=1.0, max_iter=20000 | ✓ |
| MLP | νευρωνικό | (128, 64), early stopping | ✓ |
| DeepMLP | νευρωνικό | (256, 128, 64), early stopping | ✓ |
| XGBoost | boosting | 300 δέντρα, βάθος 8, lr 0.15, subsample/colsample 0.9 | |
| LightGBM | boosting | 300 δέντρα, 64 φύλλα, lr 0.1, reg_lambda 1.0 | |
| CatBoost | boosting | 300 επαναλήψεις, βάθος 8, lr 0.15 | |
| TorchMLP | νευρωνικό | (256, 128, 64), dropout 0.2, patience 6 | ✓ |

**Πρωτόκολλο βελτιστοποίησης** ([`tuning.py`](tuning.py)): τα 4 ισχυρότερα μοντέλα (RandomForest, XGBoost, LightGBM,
CatBoost) βελτιστοποιούνται με **Optuna TPE**, 25 δοκιμές (οι πρώτες 8 τυχαίες), με στόχο το μέσο macro-F1 μιας
**στρωματοποιημένης 3-fold CV σε 30.000 γραμμές εκπαίδευσης**, σπόρος 42. Η καλύτερη διαμόρφωση επανεκπαιδεύεται σε όλο
το training split και συγκρίνεται με την προεπιλογή στο test split. Αν η επανεκπαιδευμένη διαμόρφωση είναι χειρότερη
από την προεπιλογή κατά περισσότερο από 0,05 macro-F1, το αποτέλεσμα της αναζήτησης απορρίπτεται και κρατιέται η προεπιλογή. Τα βελτιστοποιημένα μοντέλα
φέρουν την κατάληξη `__tuned`. Το `extra_studies.py` εφαρμόζει το ίδιο πρωτόκολλο στο `combined`.

## Πειράματα

| Πείραμα | Δεδομένα | Τι μετρά |
|---|---|---|
| **Ανά dataset** | κάθε dataset χωριστά, με όλα τα εγγενή χαρακτηριστικά του | απόδοση μέσα στο ίδιο δίκτυο |
| **combined** | η **ένωση** των 4 datasets στα 11 κοινά χαρακτηριστικά, **ένα μοντέλο**, τυχαίος διαχωρισμός | ένα μοντέλο για όλα τα δίκτυα (in-domain) |
| **LODO** (Leave-One-Dataset-Out) | εκπαίδευση σε 3 datasets, test στο 4ο (RandomForest, 150 δέντρα, 11 κοινά χαρακτηριστικά) | γενίκευση σε άγνωστο δίκτυο |

Το `combined` σημαίνει **εκπαίδευση στα ενωμένα datasets**. **Δεν** είναι συνδυασμός (ensemble) των
προβλέψεων πολλών μοντέλων.

Κάθε πείραμα εκτελείται για δύο εργασίες: **πολυκλασική** (κατηγορίες επιθέσεων) και **δυαδική** (Benign / Attack).

## Μετρικές

([`evaluation.py`](evaluation.py), [`validationstats.py`](validationstats.py))

* **Κύρια μετρική: macro-F1**, ο μη σταθμισμένος μέσος όρος του F1 ανά κλάση. Κάθε κλάση μετρά το ίδιο, οπότε οι
  μεγάλες κλάσεις δεν κυριαρχούν όπως στην ακρίβεια (accuracy).
* Αναφέρονται επίσης: accuracy, balanced accuracy, weighted F1, MCC, Cohen's κ, G-mean, ROC-AUC (one-vs-rest),
  log loss, precision/recall/F1 ανά κλάση, πίνακας σύγχυσης, χρόνος εκπαίδευσης και πρόβλεψης.
* **Διαστήματα εμπιστοσύνης bootstrap 95%** του macro-F1: 500 επαναδειγματοληψίες του πλήρους test split.
* **Έλεγχος McNemar** μεταξύ των δύο καλύτερων μοντέλων κάθε dataset/εργασίας (ζευγαρωτός, στις ίδιες ροές ελέγχου).
* **Τετριμμένες βάσεις αναφοράς** (most-frequent, stratified, uniform), εκπαιδευμένες στο training split, ως το κατώτατο όριο
  που πρέπει να ξεπεράσει κάθε μοντέλο.
* **Μελέτη σπόρων**: 4 σπόροι (42, 7, 13, 34) που αλλάζουν τόσο τον διαχωρισμό όσο και τους σπόρους των μοντέλων.

## Επιλογή χωρίς το test split

Η κύρια εκτέλεση αναφέρει το καλύτερο μοντέλο και τα κατώφλια των σημείων λειτουργίας **στο test split**. Οι αριθμοί αυτοί
είναι περιγραφικοί: η επιλογή στα ίδια δεδομένα στα οποία μετράται η βαθμολογία είναι αισιόδοξη. Το αυτόνομο script
[`validated_selection.py`](validated_selection.py) επαναλαμβάνει και τις δύο επιλογές **μόνο σε δεδομένα εκπαίδευσης** και τις εφαρμόζει μία φορά στο test split:

1. **Επιλογή μοντέλου με διασταυρούμενη επικύρωση.** Στρωματοποιημένη 5-fold CV σε 60.000 γραμμές εκπαίδευσης για τα RandomForest,
   ExtraTrees, XGBoost, LightGBM και CatBoost (το πρωτόκολλο του `extendedstudy.py` μέρος D). Επιλέγεται το μοντέλο
   με το υψηλότερο μέσο macro-F1 στη CV, και το macro-F1 του στο test αναφέρεται μία φορά, δίπλα στον
   νικητή που επιλέχθηκε στο test.
2. **Κατώφλια out-of-fold.** Για κάθε δυαδικό μοντέλο, η 5-fold CV στο training split δίνει μια
   βαθμολογία επίθεσης out-of-fold για κάθε ροή εκπαίδευσης (ίδιες υπερπαράμετροι με το αποθηκευμένο μοντέλο). Σε αυτές τις
   βαθμολογίες επιλέγεται το κατώφλι με τη μεγαλύτερη ανάκληση για FPR ≤ 5%, 1%, 0,1% και 0,01%, και
   εφαρμόζεται **μία φορά** στο αποθηκευμένο μοντέλο στο test split. Είναι η ιδέα του `TunedThresholdClassifierCV` του scikit-learn.
   Ένας στόχος μικρότερος από 1 / (πλήθος κανονικών ροών) δεν μπορεί να επιλεγεί ούτε να
   μετρηθεί, και αναφέρεται κενός. Πριν από κάθε μέτρηση το script ελέγχει ότι το test split που ξαναχτίστηκε
   αναπαράγει ακριβώς την καταγεγραμμένη ακρίβεια του μοντέλου.
3. **Η σημαία `evaluable`.** Κάθε γραμμή αποτελέσματος φέρει `evaluable` και `not_evaluable_reason` (από το
   `split_meta()`). Ένα κελί δεν είναι αξιολογήσιμο όταν το test split έχει μία μόνο κλάση ή περιέχει κλάσεις
   που λείπουν από την εκπαίδευση (π.χ. ο χρονολογικός διαχωρισμός του BoT-IoT). Τα μη αξιολογήσιμα κελιά δεν αναφέρονται ποτέ ως
   νικητές: οι πίνακες καλύτερων μοντέλων τα εμφανίζουν ως «μη αξιολογήσιμα» μαζί με την αιτία.

# Πώς εκτελείται το pipeline

Το [`runfull.py`](runfull.py) είναι ο **ενορχηστρωτής**. Εκτελεί κάθε φάση ως **ξεχωριστή
διεργασία** (`subprocess.Popen`), ώστε μια κατάρρευση ή διαρροή μνήμης σε μία φάση να μην
παρασύρει τις υπόλοιπες. Κάθε φάση γράφει το δικό της log
(`results/pipeline_<phase>.log`) και η κατάσταση όλων ενημερώνεται ατομικά στο
`results/run_meta.json`.

## Κύριος βραχίονας 18 φάσεων

| # | Φάση | Script | Τι κάνει |
|---|---|---|---|
| 1 | `run_all` | [`runall.py`](runall.py) | **Κρίσιμη φάση.** Φόρτωση -> προεπεξεργασία -> διαχωρισμός -> εκπαίδευση 15 μοντέλων × 2 εργασίες × 4 datasets, + μοντέλο combined, + LODO μεταξύ datasets |
| 2 | `tuning` | [`tuning.py`](tuning.py) | Optuna TPE, 25 δοκιμές × 3-fold CV × 30.000 γραμμές, για RF/XGB/LGBM/CatBoost |
| 3 | `extended_study` | [`extendedstudy.py`](extendedstudy.py) | A: στρατηγικές ανισορροπίας B: επιλογή χαρακτηριστικών C: μέγεθος δείγματος D: διασταυρούμενη επικύρωση |
| 4 | `deep_study` | [`deepstudy.py`](deepstudy.py) | Ablation ανισορροπίας, καμπύλες μάθησης, RandomizedSearchCV |
| 5 | `anomaly_study` | [`anomalystudy.py`](anomalystudy.py) | IsolationForest εκπαιδευμένο μόνο σε κανονική κίνηση (ίδιο πεδίο + άλλο πεδίο) |
| 6 | `adaptation_study` | [`adaptionstudy.py`](adaptionstudy.py) | Πόσα τοπικά δεδομένα χρειάζονται (0%, 0,5%, 1%, 2%, 5%, 10%) για προσαρμογή σε νέο δίκτυο |
| 7 | `deployment_bench` | [`deploymentbench.py`](deploymentbench.py) | Καθυστέρηση/ρυθμαπόδοση: 50.000 ροές × 3 επαναλήψεις ανά μοντέλο, μέγεθος μοντέλου στον δίσκο |
| 8 | `extended_figures` | [`make_extended_figures.py`](make_extended_figures.py) | Καμπύλες ROC/PR, heatmaps, SHAP |
| 9 | `report_assets` | [`reportassets.py`](reportassets.py) | Παράγει τα `tables.csv` και τα συνοπτικά γραφήματα |
| 10 | `validation_stats` | [`validationstats.py`](validationstats.py) | Bootstrap 95% CI (500 δείγματα), έλεγχοι McNemar, dummy baselines |
| 11 | `domain_shift_study` | [`domainshift_study.py`](domainshift_study.py) | Πόσο αναγνωρίσιμο είναι το dataset προέλευσης (source identifiability) |
| 12 | `anomaly_sweep` | [`anomalysweep.py`](anomalysweep.py) | Σάρωση contamination (0,5% → 10%) × στόχοι FPR |
| 13 | `leakage_ablation` | [`leakage_ablation.py`](leakage_ablation.py) | Αφαίρεση θυρών / συγκεντρωτικών ανά IP / χαρακτηριστικών παραθύρου — πόση από την απόδοση ήταν διαρροή |
| 14 | `operating_points` | [`operatingpoints.py`](operatingpoints.py) | Ακρίβεια διορθωμένη ως προς το βασικό ποσοστό, όγκος συναγερμών σε δίκτυο 10M ροών/ημέρα |
| 15 | `cost_sensitive` | [`costsensitive.py`](costsensitive.py) | `none` / `balanced` / `fn_cost` × κόστη 1×, 5×, 10× |
| 16 | `seed_study` | [`seedstudy.py`](seedstudy.py) | Σταθερότητα σε 4 σπόρους (42, 7, 13, 34) |
| 17 | `specialist_study` | [`specialiststudy.py`](specialiststudy.py) | A: κλιμακωτό σχήμα (gate + specialist) B: one-vs-rest specialists C: μεταφορά ανά κλάση μεταξύ datasets |
| 18 | `sanity_check` | [`sanitycheck.py`](sanitycheck.py) | Συμβουλευτική (ελέγχει μετρικές εκτός εύρους, κολλήματα, ασυνέπειες) |

## Χρονολογικός βραχίονας (με `--temporal-arm`)

Εκτελείται **μόνο για bot_iot και iot23** (τα μόνα datasets με αξιοποιήσιμη χρονοσφραγίδα)
και γράφει σε ξεχωριστούς φακέλους ώστε να μην αντικαταστήσει τον κύριο βραχίονα:
`temporal_run_all -> temporal_report_assets -> temporal_validation_stats ->
temporal_seed_study -> temporal_sanity_check`.

## Πολιτική σφαλμάτων

| Κατηγορία | Φάσεις | Συμπεριφορά σε exit code ≠ 0 |
|---|---|---|
| **Κρίσιμη** | `run_all` | το pipeline **σταματά** |
| **Συμβουλευτική** | `sanity_check`, `temporal_sanity_check` | αναφέρεται ως `findings`, δεν θεωρείται αποτυχία |
| Υπόλοιπες | όλες οι άλλες | η αποτυχία καταγράφεται, το pipeline **συνεχίζει** στην επόμενη φάση |

# Εντολές & παράμετροι

## `runfull.py`

```bash
python src/runfull.py [--fresh] [--only PHASE] [--skip PHASE (phase name)] [--smoke] [--temporal-arm]
```

| Παράμετρος | Τι κάνει |
|---|---|
| `--fresh` | Αρχειοθετεί τα τρέχοντα `results/`, `models/` και τα αντίστοιχα `_temporal` στο `archives/run_<timestamp>/` μαζί με το `manifest.json`, και μετά τα διαγράφει ώστε η εκτέλεση να ξεκινήσει από την αρχή. Χωρίς αυτό, το `runall.py` κάνει **συνέχιση**: επαναχρησιμοποιεί τα προηγούμενα αποτελέσματα όπου οι παράμετροι και το data_fingerprint ταιριάζουν. |
| `--only PHASE` | Εκτελεί **μία μόνο** φάση (π.χ. `--only tuning`). Οι άλλες κρατούν την προηγούμενη κατάστασή τους στο `run_meta.json`. |
| `--skip PHASE (phase name)` | Παραλείπει μία ή περισσότερες φάσεις (π.χ. `--skip seed_study specialist_study`). |
| `--smoke` | Παράγει συνθετικά datasets (900 γραμμές/κλάση, [`smoke_data.py`](smoke_data.py)) και ανακατευθύνει τα πάντα στα `results_smoke/`, `models_smoke/`, `archives_smoke/`. Επαληθεύει ότι όλο το pipeline τρέχει, χωρίς να χρειάζονται πραγματικά δεδομένα. |
| `--temporal-arm` | Μετά τον κύριο βραχίονα, επαναλαμβάνει BoT-IoT και IoT-23 με χρονολογικό διαχωρισμό στα `results_temporal/` και `models_temporal/`. |

Παραδείγματα:

```bash
python src/runfull.py --fresh --temporal-arm
```

```bash
python src/runfull.py --only tuning
```

```bash
python src/runfull.py --skip seed_study specialist_study leakage_ablation
```

## `runall.py` (μόνο εκπαίδευση)

```bash
python src/runall.py [dataset] [--smoke] [--no-pooled]
```

* Χωρίς ορίσματα: εκτελεί **όλα** τα datasets που βρέθηκαν στον δίσκο.
* Με ονόματα datasets: μόνο αυτά (π.χ. `python src/runall.py bot_iot iot23`).
* **`--no-pooled`**: παραλείπει τα `combined` και `LODO`. Γίνεται αυτόματα στον
  χρονολογικό βραχίονα, γιατί τα πειράματα ένωσης δεν έχουν νόημα όταν τρέχουν μόνο 2 από τα 4
  datasets.

## Εκτέλεση ίσου προϋπολογισμού (matched)

Τα αργά μοντέλα εκπαιδεύονται σε 250.000 γραμμές αντί για 60.000, για δίκαιη σύγκριση με τα γρήγορα.
Γράφει στους δικούς της φακέλους, οπότε δεν αγγίζει την κύρια εκτέλεση. Σε PowerShell:

```powershell
$env:IOT_IDS_SLOW_CAP="250000"; $env:IOT_IDS_RESULTS_DIR="results_matched"; $env:IOT_IDS_MODELS_DIR="models_matched"
python src/runall.py
python src/operatingpoints.py
python src/reportassets.py
```

Σε bash:

```bash
IOT_IDS_SLOW_CAP=250000 IOT_IDS_RESULTS_DIR=results_matched IOT_IDS_MODELS_DIR=models_matched python src/runall.py
```

## `archiverun.py` (αρχειοθέτηση)

```bash
python src/archiverun.py --list
```

```bash
python src/archiverun.py --fresh --label before_new_features
```

```bash
python src/archiverun.py --restore run_20260905_104454
```

Το `--restore` κρατά πρώτα αντίγραφο των τρεχόντων αποτελεσμάτων.

## `status.py`

```bash
python status.py
```

Διαβάζει μόνο τα `run_meta.json` και `progress.json`, δεν έχει εξαρτήσεις, και μπορεί να
εκτελεστεί με ασφάλεια ενώ το pipeline δουλεύει.

## Μεμονωμένες φάσεις

Κάθε φάση εκτελείται και μόνη της, με τα δικά της ορίσματα:

```bash
python src/tuning.py bot_iot --trials 50 --models XGBoost LightGBM --task multiclass
```

```bash
python src/extendedstudy.py ciciot2023 --parts A C --cv-cap 30000
```

```bash
python src/seedstudy.py --seeds 42 7 13 34 99 --models RandomForest XGBoost
```

## `extra_studies.py` (αυτόνομο, εκτός `runfull.py`)

Δύο επιπλέον μελέτες που δεν εκτελεί το κύριο pipeline. Εκτελείται με το χέρι, **μετά** το `run_all`, και γράφει τα πάντα στον δικό του φάκελο (`results_extra/` από προεπιλογή):

```bash
python src/extra_studies.py both
```

```bash
python src/extra_studies.py combined-tuning --trials 25 --tune-rows 30000
```

```bash
python src/extra_studies.py ciciot-merge --train-cap 250000 --slow-cap 60000
```

## `validated_selection.py` (αυτόνομο, εκτός `runfull.py`)

Επιλογή μοντέλου και κατωφλίου **χωρίς να κοιτάξει το test split** (βλ.
[Επιλογή χωρίς το test split](#επιλογή-χωρίς-το-test-split)). Διαβάζει τα `results/`,
`results_temporal/` και `models/` (μόνο ανάγνωση) και γράφει τα πάντα στο `results_validated/`.
Εκτελείται με το χέρι, **μετά** την κύρια εκτέλεση (χρειάζεται τα `all_results.json`, `extended_results.json`,
`operating_points.json` και τα αποθηκευμένα δυαδικά μοντέλα).

```bash
python src/validated_selection.py
```

```bash
python src/validated_selection.py thresholds --datasets ton_iot iot23
```

```bash
python src/validated_selection.py cv
```

```bash
python src/validated_selection.py tables
```

| Μέρος | Τι κάνει | Έξοδος |
|---|---|---|
| `thresholds` | κατώφλια out-of-fold ανά στόχο FPR για κάθε γρήγορο δυαδικό μοντέλο, εφαρμοσμένα μία φορά στο test split, σε σύγκριση με το oracle του `operating_points` | `operating_points_validated.json`, `table_operating_points_validated*.csv` |
| `cv` | επιλογή μοντέλου με 5-fold CV· τα τέσσερα μεμονωμένα datasets επαναχρησιμοποιούν το `extendedstudy.py` μέρος D, το `combined` επικυρώνεται εδώ, μετά από έλεγχο ότι το πρωτόκολλο αναπαράγει ακριβώς το μέρος D στο `ton_iot` (`--cv-check`) | `cv_combined.json`, `table_best_models_cv.csv` |
| `tables` | μόνο ξαναφτιάχνει τους πίνακες από τα παραπάνω αρχεία (δευτερόλεπτα, χωρίς εκπαίδευση): καλύτερα μοντέλα μόνο με αξιολογήσιμα κελιά (τυχαίος και χρονολογικός) και η επιλογή CV | `table_best_models_evaluable_{random,temporal}.csv`, `table_best_models_cv.csv` |

Χωρίς ορίσματα εκτελεί και τα τρία μέρη με αυτή τη σειρά.

## `confirm_tuning_seeds.py` (αυτόνομο, εκτός `runfull.py`)

Ελέγχει αν το κέρδος της βελτιστοποίησης στο `combined` είναι πραγματικό ή ένα δείγμα του θορύβου. Για κάθε σπόρο ξαναφτιάχνει
τον διαχωρισμό και το όριο εκπαίδευσης, αλλάζει τους σπόρους των μοντέλων, και εκπαιδεύει την προεπιλεγμένη και τη βελτιστοποιημένη διαμόρφωση
στον **ίδιο** διαχωρισμό· και οι δύο βαθμολογούνται στο **ίδιο** test split. Οι βελτιστοποιημένες υπερπαράμετροι είναι σταθερές,
από την αναζήτηση Optuna με σπόρο 42 του `extra_studies.py` (η αναζήτηση δεν επαναλαμβάνεται). Ο σπόρος 42 είναι η
άγκυρα: πρέπει να αναπαράγει ακριβώς το `results_extra/combined_tuning.json` (`anchor_seed42.csv`).
Εκτελείται με το χέρι, **μετά** το `extra_studies.py`, και γράφει στο `results_tuning_seeds/`.

```bash
python src/confirm_tuning_seeds.py --seeds 42 7 13 34 101 202 303 404 505 606
```

```bash
python src/confirm_tuning_seeds.py --models RandomForest --tasks multiclass
```

Ένα κέρδος χαρακτηρίζεται «επιβεβαιωμένο» όταν είναι θετικό σε κάθε σπόρο και το διάστημα 95% της μέσης
διαφοράς δεν περιέχει το μηδέν.

**Και τα δύο αυτόνομα scripts συνεχίζουν**: οι γραμμές που έχουν ήδη γραφτεί στον φάκελο εξόδου τους δεν
ξαναϋπολογίζονται. Για να ξανατρέξει κάποιο από την αρχή, μετακινήστε πρώτα τον φάκελο εξόδου του.

Συνοπτικά:

| Script | Θέσης ορίσματα | Παράμετροι |
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
| `validated_selection.py` | `parts` (thresholds cv tables) | `--datasets _` `--models _` `--cv-check DATASET` (ton_iot) |
| `confirm_tuning_seeds.py` | — | `--seeds _` (42 7 13 34) `--models _` (RandomForest XGBoost) `--tasks _` (multiclass binary) |

# Streamlit monitor

```bash
streamlit run src/trainingmonitor.py
```

Ανοίγει στο **http://localhost:8501**.

**Σελίδες:** `Run` (ζωντανή πρόοδος ανά φάση)  `Overview`,`Models`, `Tuning`,`Ablations`,`Generalisation`,`Files`.

**Πλευρική στήλη:**

* **Run**: εναλλαγή μεταξύ της τρέχουσας εκτέλεσης και οποιασδήποτε αρχειοθετημένης (`archives/run_*`)
* **Refresh**: αυτόματη ανανέωση 3–30 δευτερολέπτων ενώ τρέχει το pipeline
* **Appearance**: θέμα Light / Dark

Το dashboard είναι **μόνο για ανάγνωση**: διαβάζει τα `run_meta.json`, `progress.json`,
`all_results.json` και τα PNG του `results/figures/`. Μπορεί να ανοίξει ενώ το
pipeline τρέχει (το `checkpoint()` γράφει ατομικά μέσω προσωρινού αρχείου, οπότε
δεν διαβάζετε ποτέ μισογραμμένο JSON), και επίσης αφού τελειώσει, για την ανάλυση.

> Όσο τρέχει μια φάση `temporal`, το monitor διαβάζει αυτόματα το `progress.json` από το `results_temporal/`, ενώ το `run_meta.json` μένει πάντα στο `results/`.

# Χρόνοι εκτέλεσης

Μετρημένοι χρόνοι από το `results/run_meta.json` (εκτέλεση `pipeline_20260905_104457`,
16 GB RAM / RTX 4050 Laptop, και τα 4 datasets):

| Φάση | Διάρκεια | % της εκτέλεσης |
|---|---:|---:|
| `run_all` | 1 ώ 24 λ | 8,0% |
| `tuning` | 3 ώ 52 λ | 22,1% |
| `extended_study` | 2 ώ 41 λ | 15,3% |
| `deep_study` | 54 λ 29 δ | 5,2% |
| `anomaly_study` | 35 δ | 0,1% |
| `adaptation_study` | 4 λ 18 δ | 0,4% |
| `deployment_bench` | 19 λ 25 δ | 1,8% |
| `extended_figures` | 18 λ 28 δ | 1,8% |
| `report_assets` | 4 δ | <0,1% |
| `validation_stats` | 18 λ 13 δ | 1,7% |
| `domain_shift_study` | 6 λ 12 δ | 0,6% |
| `anomaly_sweep` | 22 δ | <0,1% |
| `leakage_ablation` | 54 λ 56 δ | 5,2% |
| `operating_points` | 11 λ 00 δ | 1,0% |
| `cost_sensitive` | 1 ώ 01 λ | 5,8% |
| `seed_study` | **3 ώ 11 λ** | 18,2% |
| `specialist_study` | 17 λ 17 δ | 1,6% |
| `sanity_check` | 1 δ | <0,1% |
| **Υποσύνολο κύριου βραχίονα** | **15 ώ 35 λ** | 89% |
| `temporal_run_all` | 26 λ 22 δ | 2,5% |
| `temporal_report_assets` | 3 δ | <0,1% |
| `temporal_validation_stats` | 8 λ 03 δ | 0,8% |
| `temporal_seed_study` | 1 ώ 21 λ | 7,7% |
| `temporal_sanity_check` | 1 δ | <0,1% |
| **Υποσύνολο χρονολογικού βραχίονα** | **1 ώ 55 λ** | 11% |
| **ΣΥΝΟΛΟ** | **17 ώ 31 λ** | 100% |

**Τρεις φάσεις καταναλώνουν το 55% του χρόνου:** `tuning` (Optuna, 25 δοκιμές × 3 folds ανά
μοντέλο ανά dataset ανά εργασία), `seed_study` (4 σπόροι × 15 μοντέλα × 2 εργασίες ×
4 datasets) και `extended_study` (4 ablations × 2 εργασίες × 4 datasets).

Το ακριβότερο μεμονωμένο μοντέλο είναι το CatBoost στο CICIoT2023 πολυκλασικό (169 s, 250.000 γραμμές), το φθηνότερο ο KNN (0,3 s· στην πράξη δεν εκπαιδεύεται, πληρώνει στην πρόβλεψη).

**Αυτόνομες μελέτες** (εκτελούνται με το χέρι, δεν μετρώνται στο παραπάνω σύνολο):

| Script | Διάρκεια |
|---|---:|
| `extra_studies.py` | ~3 ώ 30 λ |
| `validated_selection.py` | ~1 ώ |
| `confirm_tuning_seeds.py` (10 σπόροι) | ~45 λ |

# CUDA / GPU

**Το pipeline τρέχει εξ ολοκλήρου σε CPU.** Δεν υπάρχει βήμα που να *απαιτεί* GPU.

### Το μόνο σημείο που χρησιμοποιεί CUDA

Το **`TorchMLP`** ([`torchmlp.py`](torchmlp.py)), ένα MLP σε PyTorch (256→128→64, BatchNorm +
ReLU + Dropout 0.2, AdamW, early stopping με patience 6 σε validation split 10%),
τυλιγμένο σε `BaseEstimator` συμβατό με το sklearn.

```python
def pick_device(self):
    if self.device:
        return self.device
    return "cuda" if torch.cuda.is_available() else "cpu"
```
## Τρία επίπεδα εφεδρείας

1. **Δεν υπάρχει καθόλου PyTorch**: το `models_zoo.py` πιάνει το `ImportError`,
   ορίζει `HAS_TORCH = False` και το `TorchMLP` απλώς δεν μπαίνει στο zoo.
   Το pipeline τρέχει με 14 αντί για 15 μοντέλα (18 αντί για 19 διαμορφώσεις), χωρίς κανένα σφάλμα.
2. **Υπάρχει PyTorch αλλά όχι CUDA**: το `torch.cuda.is_available()` επιστρέφει `False`
   και η εκπαίδευση γίνεται σε CPU. Ίδια αποτελέσματα, περισσότερος χρόνος.
3. **Υπάρχει CUDA αλλά γεμίζει η VRAM**: το `fit()` πιάνει ρητά το `torch.cuda.OutOfMemoryError`,
   καθαρίζει την cache και ξανατρέχει την ίδια εκπαίδευση σε CPU:

```python
try:
    self.fit_on(X, yi, ti, vi, d, k, device)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    self.fit_on(X, yi, ti, vi, d, k, "cpu")
```

Η πρόβλεψη (`logits()`) υπολογίζεται σε **τμήματα των 8.192 γραμμών** και το μοντέλο
επιστρέφει στη CPU μετά από κάθε κλήση, ώστε να μην κρατά VRAM μεταξύ φάσεων.
Τα αποθηκευμένα `.joblib` περιέχουν πάντα **tensors CPU** — φορτώνονται και σε μηχάνημα χωρίς GPU.

## Τι δεν χρησιμοποιεί GPU

**Τα XGBoost, LightGBM και CatBoost τρέχουν σε CPU.** Το XGBoost χρησιμοποιεί
`tree_method="hist"` (βάσει ιστογραμμάτων, CPU) με `n_jobs=-1` παντού — **όχι** `device="cuda"`.
Ήταν συνειδητή επιλογή: δέντρα με ~250.000 γραμμές × μερικές δεκάδες χαρακτηριστικά
είναι ήδη γρήγορα σε πολυπύρηνη CPU, και έτσι τα αποτελέσματα αναπαράγονται σε
οποιοδήποτε μηχάνημα.

## Ποιες φάσεις αγγίζουν το `TorchMLP`

`run_all` · `extended_study` (ablations + καμπύλες μάθησης) · `seed_study` ·
`cost_sensitive` · `leakage_ablation` · `specialist_study` · `extended_figures` (ROC/PR).
Όλες οι υπόλοιπες είναι 100% CPU.

---

# Χρονολογικός διαχωρισμός

Το ερώτημα που απαντά: **αν εκπαιδεύσω σε παλιά κίνηση, ανιχνεύω τη νέα;**
Ο στρωματοποιημένος διαχωρισμός το κρύβει, γιατί ροές της ίδιας επίθεσης, από τα ίδια
δευτερόλεπτα καταγραφής, καταλήγουν και στο train και στο test.

## Πώς υλοποιείται

Όλη η λογική βρίσκεται στο `split()` του [`preprocessing.py`](preprocessing.py):

1. **Διατήρηση του χρόνου.** Κάθε loader καλεί το `attach_timestamp()` και αποθηκεύει τη
   χρονοσφραγίδα σε μια **δεσμευμένη στήλη `ts`** (`TS_COL`). Δέχεται δευτερόλεπτα epoch ή αναγνώσιμες ημερομηνίες, και απορρίπτει «σταθερά ρολόγια» (`nunique <= 1`) που δεν μεταφέρουν πληροφορία.
2. **Το `ts` δεν είναι ποτέ χαρακτηριστικό.** Η πρώτη ενέργεια του `split()` είναι
   `X = X.drop(columns=[TS_COL])`. **Κανένα μοντέλο δεν βλέπει τον χρόνο** (χρησιμεύει μόνο ως κλειδί ταξινόμησης).
3. **Αφαίρεση διπλοτύπων** στις στήλες ροής (εκτός των στηλών παραθύρου `w5_*` / `w10_*`), πριν από τον διαχωρισμό.
4. **Η χρονολογική τομή:**
```python
order = ts.fillna(ts.min() - 1).sort_values(kind="mergesort").index  # stable sort
cut = int(round(len(order) * (1 - TEST_SIZE)))                       # TEST_SIZE = 0.30
tr_idx, te_idx = order[:cut], order[cut:]
```

**Το παλαιότερο 70% πηγαίνει στην εκπαίδευση. Το νεότερο 30% στον έλεγχο.** Χωρίς στρωματοποίηση. Η ταξινόμηση είναι `mergesort` (σταθερή), οπότε οι χρονικές
ισοπαλίες κρατούν τη σειρά του αρχείου και ο διαχωρισμός είναι ντετερμινιστικός.

## Δύο δίχτυα ασφαλείας

| Συνθήκη | Αντίδραση |
|---|---|
| Δεν υπάρχει `ts`, ή <10 έγκυρες τιμές, ή <2 μοναδικές | **Επιστροφή σε τυχαίο** + προειδοποίηση |
| Η τομή αφήνει το train με <2 κλάσεις (ή το test με 0) | **Επιστροφή σε τυχαίο** + προειδοποίηση |

Η επιστροφή σε τυχαίο καταγράφεται πάντα, δεν συμβαίνει ποτέ σιωπηλά.

## Το `split_audit.json`

Κάθε διαχωρισμός γράφει μια εγγραφή στο `results/split_audit.json` με: το πρωτόκολλο που ζητήθηκε και αυτό που χρησιμοποιήθηκε πράγματι, μεγέθη train/test, κλάσεις σε καθένα, **κλάσεις που λείπουν
από το train** (μη μαθήσιμες), χρονικά εύρη και `time_ranges_disjoint` (απόδειξη ότι
δεν υπάρχει επικάλυψη).

## Ισχύει το ίδιο για όλα τα datasets;

**Όχι. Βασικό εύρημα του πρωτοκόλλου.** Μόνο 2 από τα 4 datasets
έχουν αξιοποιήσιμη χρονοσφραγίδα:

| Dataset | Πηγή χρόνου | Εφικτός χρονολογικός; |
|---|---|---|
| **bot_iot** | `stime` / `ltime` (Argus) | ✅ Ναι |
| **iot23** | `ts` (Zeek epoch) | ✅ Ναι |
| **ton_iot** | υπάρχει `ts`, αλλά το dataset είναι **προ-ισορροπημένο** (8 κλάσεις × 20.000) — η δειγματοληψία γραμμών κατέστρεψε τη χρονική δομή | ❌ Επιστροφή σε τυχαίο |
| **ciciot2023** | δεν διανέμεται χρονοσφραγίδα ανά ροή | ❌ Επιστροφή σε τυχαίο |

Γι' αυτό το `--temporal-arm` εκτελεί **μόνο** τα `bot_iot` και `iot23`
(`TEMPORAL_ARM_DATASETS` στο [`runfull.py`](runfull.py)) και με `--no-pooled`: τα `combined`
και LODO θα χτίζονταν από ένα σύνολο όπου τα μισά datasets θα είχαν σιωπηλά
επιστρέψει σε τυχαίο διαχωρισμό, δηλαδή θα ήταν **ένα υβρίδιο που δεν αντιστοιχεί σε κανένα
πρωτόκολλο**.

## Τι έδειξε στην πράξη

Από το `results_temporal/split_audit.json` της τελευταίας εκτέλεσης:

```
bot_iot  temporal  train 280,290  test 120,124  disjoint=True  missing_from_train=['Theft']
iot23    temporal  train 111,726  test  47,883  disjoint=True  missing_from_train=[]
```

Στο BoT-IoT η **κλάση `Theft` εμφανίζεται μόνο στο test**: όλες οι επιθέσεις κλοπής
δεδομένων καταγράφηκαν στο τελευταίο 30% της περιόδου. Είναι **αδύνατον** να μαθευτεί.
Το `split_meta()` τη σημειώνει με **`evaluable = False`** και
`not_evaluable_reason = "classes absent from train: ['Theft']"`, και κάθε γραμμή
αποτελέσματος φέρει αυτή τη σήμανση ώστε να εξαιρείται από τους πίνακες.

Αυτό είναι ακριβώς το επιχείρημα του χρονολογικού βραχίονα: ο τυχαίος διαχωρισμός έδωσε ένα
τακτοποιημένο macro-F1 για την `Theft` και ο χρονολογικός διαχωρισμός αποκάλυψε ότι
ο αριθμός αυτός ήταν προϊόν του πρωτοκόλλου, όχι της ικανότητας ανίχνευσης.

> **Προσοχή:** τα `n_train` στο audit καταγράφονται **πριν** εφαρμοστεί το
> όριο `MAX_TRAIN_ROWS = 250,000`, γι' αυτό το `bot_iot` δείχνει 280.290.

## Χειροκίνητη ενεργοποίηση (μόνο χρονολογικός)

```bash
IOT_IDS_SPLIT=temporal IOT_IDS_RESULTS_DIR=results_temporal IOT_IDS_MODELS_DIR=models_temporal python src/runall.py bot_iot iot23 --no-pooled
```

Σε PowerShell:

```powershell
$env:IOT_IDS_SPLIT="temporal"; $env:IOT_IDS_RESULTS_DIR="results_temporal"; $env:IOT_IDS_MODELS_DIR="models_temporal"; python src/runall.py bot_iot iot23 --no-pooled
```

# Πού αποθηκεύονται τα αποτελέσματα

Όλα κάτω από το **`src/`** (το `PROJECT_DIR` στο `config.py` είναι ο φάκελος του
ίδιου του `config.py`). Και οι τρεις ρίζες μπορούν να αλλάξουν με μεταβλητές περιβάλλοντος.

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
│                                     table_combined_*.csv, table_ciciot_merge*.csv, figures/, models/
├── results_validated/                validated_selection.py: operating_points_validated.json, cv_combined.json,
│                                     table_operating_points_validated*.csv, table_best_models_cv.csv,
│                                     table_best_models_evaluable_{random,temporal}.csv, split_audit.json, README.md
├── results_tuning_seeds/             confirm_tuning_seeds.py: tuning_seeds.csv, table_tuning_seeds_summary.csv,
│                                     anchor_seed42.csv, tuning_seeds.png, split_audit.json, README.md
│
└── archives/                         IOT_IDS_ARCHIVES_DIR
    └── run_<YYYYMMDD_HHMMSS>/
        ├── manifest.json             run_id, archiving time, number of files
        └── results/ models/ results_temporal/ models_temporal/
```

! = τα αρχεία που χρειάζεστε για τη συγγραφή της αναφοράς.

Τα αυτόνομα scripts ορίζουν το `IOT_IDS_RESULTS_DIR` στον δικό τους φάκελο **πριν** φορτώσουν το `config.py`,
ώστε τα αρχεία που γράφει μόνο του το pipeline (`split_audit.json`, `tuning.log`) να καταλήγουν σε εκείνον τον φάκελο
και όχι στο `results/`. Διαβάζουν το `results/` μόνο μέσω ρητών διαδρομών.

# Μεταβλητές περιβάλλοντος

Όλες διαβάζονται στο [`config.py`](config.py) και μπορούν να αλλάξουν χωρίς επεξεργασία κώδικα.

| Μεταβλητή | Προεπιλογή | Τι ορίζει |
|---|---|---|
| `IOT_IDS_DATA_DIR` | *(αυτόματη ανίχνευση)* | Φάκελος datasets (υπερισχύει των υποψηφίων) |
| `IOT_IDS_RESULTS_DIR` | `results` | Φάκελος αποτελεσμάτων |
| `IOT_IDS_MODELS_DIR` | `models` | Φάκελος μοντέλων |
| `IOT_IDS_ARCHIVES_DIR` | `archives` | Φάκελος αρχειοθέτησης |
| **`IOT_IDS_SPLIT`** | `random` | `random` \| `temporal` |
| **`IOT_IDS_SLOW_CAP`** | `60000` | Όριο γραμμών για τα «αργά» μοντέλα (KNN, LinearSVM, MLP, DeepMLP, AdaBoost, TorchMLP). Ορίστε **`250000`** για τη σύγκριση ίσου προϋπολογισμού |
| `IOT_IDS_MIN_CLASS_LOAD` | `8000` | Ελάχιστες γραμμές/κλάση κατά τη φόρτωση (ώστε οι σπάνιες επιθέσεις να μην εξαφανίζονται στο όριο) |
| `IOT_IDS_MIN_CLASS_SLOW` | `2000` | Το ίδιο για το υποδείγμα των αργών μοντέλων |
| `IOT_IDS_LOG_SCALE` | `1` | Συμπίεση `sign(x)·log1p(\|x\|)` για χαρακτηριστικά με βαριές ουρές· το `0` την απενεργοποιεί |
| `IOT_IDS_FN_COST` | `1.0` | Πόσους ψευδείς συναγερμούς «αξίζει» μία χαμένη επίθεση (`cost_sensitive`) |

Σταθερές μόνο μέσα στο αρχείο: `MAX_ROWS_PER_DATASET=400000`, `MAX_TRAIN_ROWS=250000`,
`TEST_SIZE=0.30`, `RANDOM_STATE=42`, `MIN_CLASS_ROWS=60`.

# Συνέχιση, checkpointing & αρχειοθέτηση

Το pipeline είναι σχεδιασμένο να **επιβιώνει από διακοπές**

**Checkpoint μετά από κάθε μοντέλο.** Το `checkpoint()` γράφει στο `all_results.tmp` και
μετά κάνει ατομικό `replace()`. Δεν υπάρχει ποτέ μισογραμμένο `all_results.json`.

**Συνέχιση με fingerprint.** Πριν από την εκπαίδευση, το `train_all_models()` ελέγχει:

```python
prev is not None
  and dataset != "combined"
  and (MODELS_DIR / f"{ta}.joblib").exists()
  and prev["params"] == spec.params
  and prev["data_fingerprint"] == fingerprint
```

Το `data_fingerprint` είναι ένα MD5 από: πλήθος γραμμών εκπαίδευσης + ταξινομημένα ονόματα
στηλών + κατανομή κλάσεων + την τιμή του `SLOW_MODEL_TRAIN_CAP`. **Αν
αλλάξει οτιδήποτε στα δεδομένα ή στις υπερπαραμέτρους, το μοντέλο επανεκπαιδεύεται.** Αν όχι,
επαναχρησιμοποιείται και τυπώνεται `resume <tag> previous results kept`.

Το `combined` **εξαιρείται πάντα** από τη συνέχιση, γιατί εξαρτάται από όλα τα datasets μαζί.

**Συνέχιση σε επίπεδο φάσης.** Το `runfull.py` διαβάζει το προηγούμενο `run_meta.json` και
κρατά την κατάσταση των φάσεων που δεν επιλέχθηκαν αυτή τη φορά, οπότε ένα `--only tuning` δεν σβήνει το ιστορικό των άλλων 17 φάσεων. Το `seed_study` κρατά επιπλέον το δικό του `seed_study_partial.csv`.

**Πότε να χρησιμοποιήσετε το `--fresh`:** μετά από αλλαγή στους loaders, στην προεπεξεργασία,
στο `model_zoo` ή στις σταθερές του `config.py`. Το `--fresh` αρχειοθετεί πρώτα και
διαγράφει μετά· **καμία προηγούμενη εκτέλεση δεν χάνεται ποτέ**.

# Χάρτης αρχείων

## Πυρήνας

| Αρχείο | Ρόλος |
|---|---|
| [`config.py`](config.py) | Διαδρομές, όρια, σταθερές, ταξινομία 13 κλάσεων, 11 κοινά χαρακτηριστικά |
| [`loaders.py`](loaders.py) | Οι 4 loaders + ο γενικός loader, ενοποίηση ετικετών· χαρακτηριστικά παραθύρου, `attach_timestamp`, `localize` |
| [`preprocessing.py`](preprocessing.py) | `build_xy`, `split` (random/temporal), `make_preprocessor`, `split_meta`, split audit |
| [`models_zoo.py`](models_zoo.py) | Τα `ModelSpec` (11 βασικά + 4 προαιρετικά), `LabelEncodedClassifier`, `build_with_class_weight` |
| [`torchmlp.py`](torchmlp.py) | MLP σε PyTorch με API του sklearn, early stopping, CUDA, εφεδρεία σε CPU |
| [`evaluation.py`](evaluation.py) | `evaluate_model`: 20+ μετρικές, πίνακας σύγχυσης, ROC/PR, feature importance |
| [`progress.py`](progress.py) | Το `progress.json` που τροφοδοτεί το monitor |

## Ενορχήστρωση

| Αρχείο | Ρόλος |
|---|---|
| [`runfull.py`](runfull.py) | Ο ενορχηστρωτής των 18 (+5) φάσεων |
| [`runall.py`](runall.py) | Η κύρια εκπαίδευση: ανά dataset, combined, LODO |
| [`archiverun.py`](archiverun.py) | archive / list / restore / clear |
| [`sanitycheck.py`](sanitycheck.py) | Έλεγχος ακεραιότητας (exit 1 αν βρει προβλήματα) |
| [`smoke_data.py`](smoke_data.py) | Συνθετικά datasets 900 γραμμών/κλάση για το `--smoke` |

## Μελέτες

[`tuning.py`](tuning.py) [`extendedstudy.py`](extendedstudy.py) [`deepstudy.py`](deepstudy.py) [`anomalystudy.py`](anomalystudy.py) [`anomalysweep.py`](anomalysweep.py) [`adaptionstudy.py`](adaptionstudy.py) [`deploymentbench.py`](deploymentbench.py) [`domainshift_study.py`](domainshift_study.py) [`leakage_ablation.py`](leakage_ablation.py) [`operatingpoints.py`](operatingpoints.py) [`costsensitive.py`](costsensitive.py) [`seedstudy.py`](seedstudy.py) [`specialiststudy.py`](specialiststudy.py) [`validationstats.py`](validationstats.py)

## Αυτόνομες μελέτες (εκτός `runfull.py`)

| Αρχείο | Ρόλος |
|---|---|
| [`extra_studies.py`](extra_studies.py) | Optuna στο `combined`, συγχώνευση DoS/DDoS στο CICIoT2023 → `results_extra/` |
| [`validated_selection.py`](validated_selection.py) | Επιλογή μοντέλου με CV, κατώφλια out-of-fold, πίνακες καλύτερων μοντέλων μόνο με αξιολογήσιμα κελιά → `results_validated/` |
| [`confirm_tuning_seeds.py`](confirm_tuning_seeds.py) | Προεπιλογή έναντι βελτιστοποίησης στο `combined` σε 10 σπόρους → `results_tuning_seeds/` |

## Παραγωγή υλικού αναφοράς

[`reportassets.py`](reportassets.py) (πίνακες CSV) [`make_extended_figures.py`](make_extended_figures.py)
(ROC/PR, heatmaps, SHAP) [`trainingmonitor.py`](trainingmonitor.py) (Streamlit dashboard)

# Αναπαραγωγιμότητα

**Σταθερός σπόρος παντού.** Το `RANDOM_STATE = 42` περνά σε κάθε estimator, κάθε `train_test_split`,
κάθε `sample()`, στο `torch.manual_seed` και στο `np.random.seed`. Το `seed_study`
μετρά ρητά πόσο μετακινούνται τα αποτελέσματα με τους σπόρους 42 / 7 / 13 / 34.

**Καμία διαρροή ετικέτας.** Οι στήλες `category` / `subcategory` / `attack` /
`detailed-label` αφαιρούνται αμέσως μόλις παραχθεί το `label`. Διευθύνσεις IP,
`flow_id`, `uid` και χρονοσφραγίδες αφαιρούνται (`GENERIC_DROP`)· η χρονοσφραγίδα
επιβιώνει μόνο στο `ts`, που το `split()` αφαιρεί πριν το μοντέλο δει οτιδήποτε.

**Ο προεπεξεργαστής είναι μέρος του pipeline.** `Pipeline([("pre", ...), ("clf", ...)])`·
το imputation, η λογαριθμική συμπίεση και η κλιμάκωση μαθαίνονται **μόνο** από το τμήμα εκπαίδευσης, και
χτίζεται νέος προεπεξεργαστής ανά μοντέλο. Το αποθηκευμένο `.joblib` περιέχει
**ολόκληρο** το pipeline, οπότε η εγκατάσταση δεν χρειάζεται να ξαναχτίσει τίποτα.

**Αφαίρεση διπλοτύπων πριν από τον διαχωρισμό**, στις στήλες ροής (πανομοιότυπες ροές δεν μπορούν να βρίσκονται ταυτόχρονα σε train και test).

**Έλεγχος σύγκλισης.** Κάθε εγγραφή φέρει `n_iter` και `converged`, ώστε να φαίνεται
αν ένας επαναληπτικός αλγόριθμος σταμάτησε επειδή συνέκλινε ή επειδή εξαντλήθηκε ο προϋπολογισμός.

**Εντιμότητα στις μετρικές.** Κάθε γραμμή αποτελέσματος φέρει `split_mode`,
`evaluable`, `not_evaluable_reason`, `classes_missing_from_train`, `classes_undertrained`,
`fully_learnable`, `train_min_class_rows`, `capped` και `train_cap`.

Το **`leakage_ablation`** ποσοτικοποιεί πόση από την απόδοση οφειλόταν στους αριθμούς θυρών,
στα συγκεντρωτικά ανά IP του BoT-IoT και στα χαρακτηριστικά πλαισίου παραθύρου (πληροφορία που δεν γενικεύεται σε άλλο δίκτυο).

**Επιλογή χωρίς το test split.** Το `validated_selection.py` επαναλαμβάνει την επιλογή μοντέλου (5-fold CV) και την
επιλογή κατωφλίου (βαθμολογίες out-of-fold) μόνο σε δεδομένα εκπαίδευσης, και ελέγχει πριν από κάθε μέτρηση ότι
αναπαράγει ακριβώς την καταγεγραμμένη ακρίβεια κάθε αποθηκευμένου μοντέλου. Βλ.
[Επιλογή χωρίς το test split](#επιλογή-χωρίς-το-test-split).

**Έλεγχος σπόρων για το κέρδος της βελτιστοποίησης.** Το `confirm_tuning_seeds.py` επαναλαμβάνει τη σύγκριση προεπιλογής–βελτιστοποίησης στο
`combined` σε 10 σπόρους· ο σπόρος 42 αναπαράγει ακριβώς το `results_extra/combined_tuning.json`. Αποτέλεσμα:
RandomForest πολυκλασική +0,0083 ± 0,0040 macro-F1, καλύτερο σε 10/10 σπόρους.

# Γνωστοί περιορισμοί

* **BoT-IoT:** μόνο 141 κανονικές ροές στο test split. Τιμές FPR κάτω από ~0,7% δεν μπορούν να μετρηθούν
  αξιόπιστα, και τα δυαδικά αποτελέσματα στηρίζονται σε πολύ λίγα κανονικά δείγματα.
* **Χρονολογικός διαχωρισμός του BoT-IoT:** το test split περιέχει μόνο DDoS και Theft, η Theft δεν εμφανίζεται
  στην εκπαίδευση, και δεν υπάρχει Benign. Δεν μπορεί να αξιολογηθεί· κάθε γραμμή σημειώνεται `evaluable = False`.
* **combined:** ο τυχαίος διαχωρισμός δίνει in-domain αποτέλεσμα. Το dataset προέλευσης μιας ροής αναγνωρίζεται
  με ακρίβεια 99,9% από τα 11 κοινά χαρακτηριστικά, οπότε η γενίκευση σε νέο δίκτυο κρίνεται από το
  LODO, όχι από το `combined`.
* **Επιλογή μοντέλου με CV:** καλύπτει τα 5 ισχυρότερα από τα 15 μοντέλα, σε 60.000 γραμμές εκπαίδευσης. Τα κατώφλια
  out-of-fold καλύπτουν μόνο τα γρήγορα μοντέλα: τα αργά εκπαιδεύτηκαν σε υποδείγμα 60.000 γραμμών, οπότε
  η επανεκπαίδευσή τους σε τμήματα όλου του training split δεν θα ήταν το ίδιο μοντέλο.
* **Έλεγχος σπόρων της βελτιστοποίησης:** οι βελτιστοποιημένες υπερπαράμετροι είναι αυτές της αναζήτησης με σπόρο 42. Ο έλεγχος επιβεβαιώνει
  ότι αυτή η διαμόρφωση είναι σταθερά καλύτερη, όχι ότι η αναζήτηση δίνει πάντα κέρδος.
* **Αποθηκευτικός χώρος:** τα μοντέλα και τα αποτελέσματα πιάνουν αρκετά GB. Ορισμένα αρχεία `.joblib` (RandomForest) ξεπερνούν τα 100 MB.

# Αντιμετώπιση προβλημάτων
**`No datasets found. Place them under the dataset dir (see config.py)`**
Το `dataset_dir()` δεν βρήκε τίποτα. Ελέγξτε με:

```bash
python -c "import sys; sys.path.insert(0,'src'); from loaders import available_datasets; from config import dataset_dir; print(dataset_dir()); print(available_datasets())"
```

Τρεις πιθανές αιτίες, με αυτή τη σειρά συχνότητας:

1. **Ο φάκελος `datasets/` δεν είναι στη σωστή θέση** (δίπλα στο `src/`
   ή μέσα στο `src/`). Διαφορετικά ορίστε το `IOT_IDS_DATA_DIR`.
2. **Ο φάκελος βρέθηκε αλλά η λίστα είναι κενή**: Τα αρχεία δεν έχουν αναγνωρίσιμο όνομα. Μετονομάστε τα σε `bot_iot.zip`, `ciciot2023.zip`, `iot_23.tar.gz`, `ton_iot.csv`.

**Το BoT-IoT φορτώνει 0 γραμμές**
Το ZIP δεν περιέχει τα CSV `Full5pc`. Ο loader αγνοεί σκόπιμα τα `10-best features`· χρειάζεστε τα `5%/All features/UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`.

**Το monitor δείχνει «stalled»**
Το `sanitycheck.py` θεωρεί κόλλημα τα **70 λεπτά** χωρίς ενημέρωση του `progress.json`.
Οι φάσεις `tuning` και `seed_study` μπορούν θεμιτά να μένουν σιωπηλές για πολύ ώρα σε ένα
βαρύ μοντέλο. Επιβεβαιώστε με `python status.py` ή με το πιο πρόσφατο `pipeline_<phase>.log`.

**`PermissionError` στο `all_results.json` (Windows)**
Το Streamlit κρατά το αρχείο ανοιχτό. Το `checkpoint()` κάνει ήδη 5 προσπάθειες με
παύση 0,3 s. Αν επιμένει, κλείστε προσωρινά το dashboard.

**Λείπει ένα μοντέλο από τα αποτελέσματα**
Ελέγξτε το `pipeline_run_all.log` για `Exception: skipping ...` ή `MemoryError: skipping ...`.
Κάθε αποτυχία μοντέλου καταγράφεται και το pipeline προχωρά.

**Το `TorchMLP` δεν εμφανίζεται πουθενά**
Το PyTorch δεν είναι εγκατεστημένο. Ελέγξτε με
`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

**`OSError: not enough free space ... to copy`**
Το `localize()` προσπαθεί να αντιγράψει ένα dataset από άλλο δίσκο στο
`datasets/local_cache/` και χρειάζεται μέγεθος αρχείου +20%.

**Μια αυτόνομη μελέτη δεν ξαναϋπολογίζει τίποτα**
Τα `extra_studies.py`, `validated_selection.py` και `confirm_tuning_seeds.py` συνεχίζουν από τα αρχεία που υπάρχουν ήδη
στον φάκελο εξόδου τους. Μετακινήστε τον φάκελο (ή διαγράψτε το συγκεκριμένο αρχείο) για να ξανατρέξει από την αρχή.

**Θέλω να επιβεβαιώσω ότι όλα λειτουργούν πριν δεσμεύσω 20 ώρες**

```bash
python src/runfull.py --smoke
```

Εκτελεί όλο το pipeline σε συνθετικά δεδομένα, σε λιγότερο χρόνο, χωρίς να αγγίζει τα `results/` και `models/`.
