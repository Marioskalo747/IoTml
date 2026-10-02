# Detection demo 

A self-contained folder that shows the whole chain of a detection system:


packet capture -> aggregation into flows -> 11 features -> model -> decision


The same app (`app.py`) runs on two data folders:

| Set | Folder | Flows from | Models |
|---|---|---|---|
| **All 4 datasets** | `demo/` | BoT-IoT, TON-IoT, CICIoT2023, IoT-23 | the 3 `combined` models: CatBoost, XGBoost, TorchMLP (11 common features) |
| **TON-IoT + IoT-23** | `demo/iot23_toniot/` | TON-IoT, IoT-23 | 4 per-dataset models: XGBoost and CatBoost for each (native Zeek fields) |

## Start

From the project folder, where `src/` and `demo/` are.

All 4 datasets:

```bash
streamlit run demo/app.py
```

TON-IoT + IoT-23:

```bash
streamlit run demo/app.py -- --data demo/iot23_toniot
```

The `--` before `--data` is required: without it Streamlit tries to read `--data` as its own flag.

Nothing has to be generated first: the CSVs and the models of both sets are included in [Google Drive](https://drive.google.com/drive/folders/1fAUY85S_PFuEEd2b6peL7fIt_bXOBQPr?usp=drive_link). The app opens at **http://localhost:8501**.

## What it contains

| file | what it is |
|---|---|
| `app.py` | the interface (5 tabs), shared by both sets |
| `flowize.py` | the aggregation of packets into flows|
| `make_traffic.py` | rebuilds `packets.csv` and `flows.csv` of the 4-dataset set from `src/results/common_cache` (it does not need the original datasets) with the same seed it produces exactly the same sample, with `--seed N` a different one |
| `packets.csv` | 4,704 packets, columns like a Wireshark export |
| `flows.csv` | 264 flows with the true class (12 classes × 22) |
| `models/` | start with three trained models the `combined` ones of `src/models_matched`: CatBoost, XGBoost, TorchMLP |
| `iot23_toniot/packets.csv` `flows.csv` | same format: 2,109 packets, 269 flows |
| `iot23_toniot/native_ton_iot.csv` · `native_iot23.csv` | the original fields of each flow (22 and 25), as the models saw them in the test |
| `iot23_toniot/models/` | `ton_iot__XGBoost`, `ton_iot__CatBoost`, `iot23__XGBoost`, `iot23__CatBoost` copies of `src/models_matched/<dataset>__multiclass__*.joblib` |
| `iot23_toniot/make_subset.py` | rebuilds everything in `iot23_toniot/` from the original datasets (IoT-23 takes ~5 minutes) |

## Folder structure
Copy the Drive files into the **same places** of the tree.
```
IoTml/                                 
│
├── src/                                (these are needed by the demo)
│   ├── config.py                       
│   ├── preprocessing.py                preprocessor stored inside the .joblib
│   ├── models_zoo.py                   LabelEncodedClassifier (XGBoost)
│   ├── torchmlp.py                     TorchMLP
│   └── requirements.txt                
│
└── demo/
    ├── Readme.md                       
    ├── app.py                             
    ├── flowize.py                       
    ├── make_traffic.py                    
    ├── packets.csv                     4,704 packets
    ├── flows.csv                       264 flows with the true class
    ├── models/                         the 3 combined models
    │   ├── CatBoost.joblib
    │   ├── XGBoost.joblib
    │   └── TorchMLP.joblib
    │
    └── iot23_toniot/                   TON-IoT + IoT-23
        ├── make_subset.py              (optional) rebuilds set 2
        ├── packets.csv                 2,109 packets
        ├── flows.csv                   269 flows
        ├── native_ton_iot.csv          native Zeek fields
        ├── native_iot23.csv            native Zeek fields
        └── models/                     the 4 per-dataset models
            ├── ton_iot__XGBoost.joblib
            ├── ton_iot__CatBoost.joblib
            ├── iot23__XGBoost.joblib
            └── iot23__CatBoost.joblib
```

The demo does **not** need the datasets or `src/results/`. The saved models contain objects from
`src/`, so the four `src` files above must be present. Without `torch`, TorchMLP is skipped with a
warning and the app runs with the other models.

```bash
pip install -r src/requirements.txt
streamlit run demo/app.py
streamlit run demo/app.py -- --data demo/iot23_toniot
```

IoTml/                                   <- ο φάκελος του project (εδώ τρέχουν οι εντολές)
│
├── src/
│   ├── config.py                        🟦 σταθερές
│   ├── preprocessing.py                 🟦 ο preprocessor μέσα στα .joblib
│   ├── models_zoo.py                    🟦 LabelEncodedClassifier (XGBoost)
│   ├── torchmlp.py                      🟦 το TorchMLP
│   └── requirements.txt                 🟩 εκδόσεις βιβλιοθηκών
│
└── demo/
    ├── app.py                           🟦 η εφαρμογή Streamlit (5 καρτέλες)
    ├── app2.py                          🟦 ίδια εφαρμογή, αισθητικές διαφορές
    ├── live.py                          🟦 ζωντανή προβολή πακέτα → ροές → απόφαση
    ├── flowize.py                       🟦 συνάθροιση πακέτων σε ροές
    ├── make_traffic.py                  🟦 (προαιρετικό) ξαναφτιάχνει τα CSV του σετ 1
    ├── README.md  README_el.md          🟩 οδηγίες (EN / EL)
    ├── packets.csv                      🟩 4.704 πακέτα
    ├── flows.csv                        🟩 264 ροές με την αληθινή κλάση
    ├── models/                          🟩 τα 3 μοντέλα του combined
    │   ├── CatBoost.joblib
    │   ├── XGBoost.joblib
    │   └── TorchMLP.joblib
    │
    └── iot23_toniot/                    σετ 2: TON-IoT + IoT-23
        ├── make_subset.py               🟦 (προαιρετικό) ξαναφτιάχνει το σετ 2
        ├── README.md                    🟩
        ├── packets.csv                  🟩 2.109 πακέτα
        ├── flows.csv                    🟩 269 ροές
        ├── native_ton_iot.csv           🟩 αρχικά πεδία Zeek
        ├── native_iot23.csv             🟩 αρχικά πεδία Zeek
        └── models/                      🟩 4 μοντέλα ανά σύνολο
            ├── ton_iot__XGBoost.joblib
            ├── ton_iot__CatBoost.joblib
            ├── iot23__XGBoost.joblib
            └── iot23__CatBoost.joblib

## Rebuilding the data (optional)

Only needed to draw a different sample or after a new run, the demo itself does not need it.

All 4 datasets needs `src/results/common_cache/` (from `src/runall.py`) and `src/results_matched/all_results.json`:

```bash
python demo/make_traffic.py
```

```bash
python demo/make_traffic.py --seed 7 --out demo/other_sample
```

TON-IoT + IoT-23 needs the original datasets, `src/models_matched/` and `src/results_matched/all_results.json`:

```bash
python demo/iot23_toniot/make_subset.py
```

A folder made with `--out` is opened the same way: `streamlit run demo/app.py -- --data demo/other_sample` (it needs its own `models/`).

## The five tabs

1. **Capture**: the traffic as a packet analyzer would show it
2. **Flows**: what the aggregation produces, with an explanation of each feature (and, in the TON-IoT + IoT-23 set, the native fields of each dataset)
3. **Predictions**: what each model said about each flow, correct or wrong
4. **Performance**: accuracy, speed, per-class breakdown, confusion matrix
5. **One flow step by step**: its packets, its features, each model's decision

# Set 1 (4 datasets)

## Where the traffic comes from

The packets are synthetic but statistics aren't. Every flow is a real record from the pooled test split, the 30% that no model saw during training. The flow
is decomposed into the packets that would have produced it, so that the demo shows the whole
chain without inventing statistics that no model has encountered.

**Verification that the flows are unseen.** The test split is rebuilt with the same
concatenation order, the same deduplication and the same seed as `runall.py`. The reconstruction
reproduces the recorded confusion matrices of all three models in
`results_matched/all_results.json`, and the 264 flows are all in the test and
none in the train. If the cache does not match the run that trained the models,
`make_traffic.py` stops instead of producing a sample that is not held-out.

**Fidelity verification.** The aggregation recovers **264/264** of `duration`, `total_pkts`,
`total_bytes`, `avg_pkt_size`, the protocol and the ports, i.e. nine of the eleven
features (`duration` with only a rounding deviation below 10⁻⁶).
Predictions on the reconstructed flows compared with the original features:

| model | same predictions | accuracy (original -> reconstructed) |
|---|---|---|
| CatBoost | 262/264 | 80.3% -> 80.3% |
| XGBoost | 256/264 | 83.0% -> 82.2% |
| TorchMLP | 263/264 | 73.5% -> 73.1% |

The deviations are in `pkt_rate`/`byte_rate` and **86 of 86 concern CICIoT2023 flows**,
the other three sets are recovered 100%. The cause is documented: that set does not
compute `pkt_rate` from the packets but keeps its own `rate` field, which no
packet capture can reproduce. In 3 of these flows `byte_rate` also deviates,
by ≤0.01%: they last a few microseconds and the time is written in
nanoseconds, as in Wireshark.

**Representativeness.** `MAX_PKTS = 120` limit. At 40 packets the
sample covered just **0.3%** of the Mirai flows, with a median of 21 packets against 100 in the true
distribution, and 67% of the DDoS. These classes were not represented. At 120 the
coverage is **99.8–100% in every class** and the sample median equals the true one in all 12 classes.

**No identifier leakage.** The models accept only the eleven features.
The IPs, the timestamps and `flow_id` exist in `packets.csv` only for human
viewing. The addresses are generated from a hash of the flow's position in the shuffled sample, so
by construction they are unrelated to the class. This was also verified quantitatively: a classifier that sees
**only** the IPs achieves 0.0873 against a chance level of 0.0833, and with only `flow_id` 0.0721.

**Bytes.** The `Payload` column keeps the bytes as each set defines them: payload in
TON-IoT, bytes including headers in BoT-IoT, IoT-23 and CICIoT2023.

## What it shows and what it does not

**It shows** that the models work: they catch 87–91% of the attacks in this
balanced sample, in tens of microseconds per flow.

**It does not show** that they are ready for deployment. The sample is balanced (22 flows per
class). On the full test split the three models flag **5.5–9.5%** of the
benign traffic as attacks (in the sample: 3 of the 22 benign flows):

| model | false alarms on Benign | attacks caught | PPV at 0.1% prevalence | PPV at 1% prevalence |
|---|---|---|---|---|
| XGBoost | 9.5% | 96.8% | 0.010 | 0.094 |
| CatBoost | 7.7% | 96.3% | 0.012 | 0.112 |
| TorchMLP | 5.5% | 94.9% | 0.017 | 0.147 |

A real network has an attack prevalence of around 0.1%. There roughly **98–99 out of 100** alerts are false.

# Set 2 — TON-IoT + IoT-23 with the per-dataset models

Flows only from TON-IoT and IoT-23, and the four models that were trained **only** on one
of the two sets.

## Why the `native_*.csv` are needed

The per-dataset models don't accept the 11 common features. They accept the Zeek fields:
`conn_state`, `service`, bytes and packets per direction, the `dns_*`/`ssl_*`/`http_*` in TON-IoT,
the windowed `w5_*`/`w10_*` in IoT-23. None of these exists in the synthetic packets.
The app reads them from `native_<dataset>.csv`, and each model is scored only on the flows of its own
set (on the others it shows "—"). A TON-IoT model cannot read an IoT-23 flow.

## Classes

| set | classes | flows |
|---|---|---|
| TON-IoT | Benign, Botnet, BruteForce, DDoS, DoS, Injection, Ransomware, Recon, Spoofing_MITM, Web | 20 each |
| IoT-23 | Benign, Botnet, Recon | 20 each |
| IoT-23 | DDoS | only 9 |

The IoT-23 models know only these four classes. The study's IoT-23 has no
others (the rare ones were merged or removed at load time).

**IoT-23's DDoS is not represented.** In 95% and more of these flows Zeek has 0 packets,
0 bytes and duration 0. A flow without packets cannot become a packet list, so the 9 included are
the minority that has packets (0.4% of the DDoS in the test).

## Verifications (`make_subset.py`)

* **Unseen flows.** The split of each set is rebuilt and the script stops if it does not reproduce
  exactly the recorded test size and the accuracy of all four models in
  `results_matched/all_results.json` (TON-IoT 53,240 rows, 0.986983 / 0.986026, IoT-23
  47,883, 0.999415 / 0.999269). Every flow is also in the pooled test split, so
  not even the combined models have seen it. 
* **Packets.** The aggregation recovers 269/269 of all 11 features.
* **CSV.** The predictions from the `native_*.csv` are identical to those from the data
  in memory (200/200 and 69/69).

# Dependency on `src`

The saved models contain transformers that point to `src/preprocessing.py`,
so `app.py` adds `../src` to the path. One source of truth, if the
preprocessing code were copied here, it could drift from the one that trained the models.
If an optional library is missing (e.g. `torch` for TorchMLP), the app
skips that model with a warning and runs with the rest.
