#Libraries
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
HERE = Path(__file__).resolve().parent
# the saved models contain transformers that point to preprocessing.py
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import joblib
from flowize import packets_to_flows, FEATURES, model_columns, read_native, native_input

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=HERE,
                 help="folder with packets.csv, flows.csv, models/ and optionally native_<dataset>.csv")
DATA = ap.parse_known_args()[0].data.resolve()
NA = "—"   # prediction for a flow the model cannot read

st.set_page_config(page_title="IoT IDS · Demo", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
  .block-container{padding-top:2rem;max-width:1400px}
  h1,h2,h3{letter-spacing:-.01em}
  .lead{color:#6b7280;font-size:.95rem;margin-top:-.4rem}
  .card{border:1px solid rgba(128,128,128,.25);border-radius:6px;padding:14px 16px;height:100%}
  .card .k{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:#6b7280}
  .card .v{font-size:1.7rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.2}
  .ok{color:#0f6b52}.bad{color:#9a3412}
  code{font-size:.85em}
  .stDataFrame{font-variant-numeric:tabular-nums}
</style>""", unsafe_allow_html=True)

#read the capture (cached by Streamlit)
@st.cache_data
def load_capture(data: Path):
    pk = pd.read_csv(data / "packets.csv")
    fl = pd.read_csv(data / "flows.csv").set_index("flow_id")
    native = {p.stem.removeprefix("native_"): read_native(p) for p in sorted(data.glob("native_*.csv"))}
    return pk, fl, native

# Aggregate packets into flows
@st.cache_data
def aggregate(pk: pd.DataFrame):
    t0 = time.perf_counter()
    fl = packets_to_flows(pk)
    return fl, time.perf_counter() - t0

# Load the models from the folder, skipping those that fail to load
@st.cache_resource
def load_models(data: Path, names):
    out, failed = {}, {}
    for n in names:
        p = data / "models" / f"{n}.joblib"
        if not p.exists():
            continue
        try:
            out[n] = joblib.load(p)
        except Exception as e:  # e.g. an optional library is missing (torch, catboost, xgboost)
            failed[n] = f"{type(e).__name__}: {e}"
    return out, failed

#the 11 features from the packets or its own dataset's fields
def model_input(m, X, native):
    cols = model_columns(m)
    if cols is None or set(cols) <= set(FEATURES):
        return X, None
    for ds, tbl in native.items():
        if set(cols) <= set(tbl.columns):
            return native_input(m, tbl.loc[tbl.index.intersection(X.index)]), ds
    raise KeyError("no native_<dataset>.csv in the folder has its fields")

#Run all models returning predictions, confidence, which flows were read, and timing
def predict_all(models, X, native):
    res = {}
    for name, m in models.items():
        try:
            Xm, ds = model_input(m, X, native)
        except KeyError as e:
            st.warning(f"Model {name} is skipped: {e}.")
            continue
        pos = X.index.get_indexer(Xm.index)
        t0 = time.perf_counter()
        p = np.asarray(m.predict(Xm)).ravel().astype(str)
        dt = time.perf_counter() - t0  #inference time
        pred = np.full(len(X), NA, dtype=object)  #start with NA
        pred[pos] = p   #fill in the flows the model could read
        conf = np.full(len(X), np.nan)
        try:
            conf[pos] = m.predict_proba(Xm).max(axis=1) #confidence of the predicted class
        except Exception:
            pass  #model without predict_proba
        mask = np.zeros(len(X), bool) 
        mask[pos] = True
        res[name] = {"pred": pred.astype(str), "conf": conf, "mask": mask, "reads": ds,
                     "seconds": dt, "us_per_flow": dt / max(len(Xm), 1) * 1e6}
    return res

#data
packets, truth, native = load_capture(DATA)
flows, agg_s = aggregate(packets)  #11 featureds 
labels = truth.loc[flows.index, "label"].astype(str).values  #true labels 

AVAILABLE = sorted(p.stem for p in (DATA / "models").glob("*.joblib"))

st.sidebar.markdown("### Models")
chosen = st.sidebar.multiselect("Up to four", AVAILABLE, default=AVAILABLE[:4],
                                max_selections=4, label_visibility="collapsed")
st.sidebar.markdown("### Capture")
st.sidebar.markdown(f"<div class='lead'>{len(packets):,} packets<br>{len(flows):,} flows"
                    f"<br>{truth['label'].nunique()} classes</div>", unsafe_allow_html=True)
st.sidebar.markdown("### Origin")
st.sidebar.dataframe(truth["source_dataset"].value_counts().rename("flows"), width="stretch")  #how many flows per dataset

st.title("From packet to decision")
st.markdown("<p class='lead'>The chain a real detection system needs, " 
            "on flows that no model saw during training.</p>", unsafe_allow_html=True)

#Early stopping if no model was chosen or none could be loaded
if not chosen:
    st.warning("Pick at least one model from the sidebar.")
    st.stop()

models, failed = load_models(DATA, chosen)
for name, err in failed.items():
    st.warning(f"Model {name} failed to load and is skipped ({err}).")
if not models:
    st.error("None of the selected models could be loaded.")
    st.stop()
X = flows[FEATURES] #feature matrix of training column
results = predict_all(models, X, native)
if not results:
    st.error("None of the selected models can read these flows.")
    st.stop()

#Five stages of pipeline
tab_cap, tab_flow, tab_pred, tab_score, tab_one = st.tabs(
    ["1 · Capture", "2 · Flows", "3 · Predictions", "4 · Performance", "5 · One flow step by step"])

#1 capture
with tab_cap:
    #raw packets as a packet analyzer would see them
    st.subheader("The traffic as a packet analyzer would show it")
    st.markdown("<p class='lead'>Same columns as a Wireshark export. <code>Payload</code> "
                "is the bytes the source dataset counts for each packet: payload only in "
                "TON-IoT, bytes including headers in BoT-IoT, IoT-23 and CICIoT2023. "
                "<code>Length</code> adds a nominal frame header, for display only.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 2])
    protos = ["all"] + sorted(packets["Protocol"].unique())
    fp = c1.selectbox("Protocol", protos)
    fid = c2.selectbox("Flow", ["all"] + [str(i) for i in flows.index])
    view = packets
    if fp != "all":
        view = view[view["Protocol"] == fp]
    if fid != "all":
        view = view[view["flow_id"] == int(fid)]
    c3.markdown(f"<div class='lead' style='padding-top:1.9rem'>{len(view):,} packets</div>", unsafe_allow_html=True)
    st.dataframe(view[["No.", "Time", "Source", "Destination", "Protocol", "Length", "Payload", "Info"]], width="stretch", height=460, hide_index=True)

#2 flows 
with tab_flow:
    #aggregated flows as a flow exporter would see them
    st.subheader("After aggregation: the eleven features")
    st.markdown(f"<p class='lead'>{len(packets):,} packets became {len(flows):,} flows in "
                f"{agg_s*1000:.0f} ms. This is exactly what a flow exporter (Zeek, nfstream) does "
                f"in a real system.</p>", unsafe_allow_html=True)
    show = flows.copy()
    show.insert(0, "true class", labels)
    show.insert(1, "source", truth.loc[flows.index, "source_dataset"].values)
    st.dataframe(show, width="stretch", height=440)
    with st.expander("How each one is computed"):
        #Documentation table
        st.markdown("""
| feature | computation |
|---|---|
| `duration` | last − first timestamp of the flow |
| `total_pkts` | number of packets |
| `total_bytes` | sum of the `Payload` column, with each dataset's definition (payload only in TON-IoT, with headers in the other three) |
| `pkt_rate` · `byte_rate` | divided by the duration — **zero** if the duration is zero |
| `avg_pkt_size` | bytes ÷ packets |
| `proto_tcp/udp/icmp` | one-hot of the protocol |
| `src_port` · `dst_port` | from the **first** packet, which sets the direction |

Same computation as the study's loaders.

The byte definition is not uniform across the four datasets and this is a limitation
of the study itself with the demo simply reproduces it.""")
    #show original Zeek fields
    for ds, tbl in native.items():
        st.markdown(f"#### The original fields of {ds}")
        st.markdown(f"<p class='lead'>{tbl.shape[1]} Zeek fields per flow, as seen by the models "
                    f"trained only on {ds}. They are not derived from the packets above.</p>",
                    unsafe_allow_html=True)
        st.dataframe(tbl, width="stretch", height=300)
#3 predictions
with tab_pred:
    #one row per flow 
    st.subheader("What each model said about each flow")
    tbl = pd.DataFrame({"true": labels}, index=flows.index)
    for name, r in results.items():
        tbl[name] = r["pred"]
        #NA for flows the model cannot read, correct/wrong for the rest
        tbl[f"{name} OK"] = np.where(~r["mask"], NA, np.where(r["pred"] == labels, "correct", "wrong"))
    only_wrong = st.checkbox("Only flows that at least one model got wrong", value=False)
    if only_wrong:
        m = np.zeros(len(tbl), bool)
        for name, r in results.items():
            m |= r["mask"] & (r["pred"] != labels)
        tbl = tbl[m]
    st.dataframe(tbl, width="stretch", height=460)
    st.markdown(f"<p class='lead'>{len(tbl):,} flows shown. “{NA}”: a flow from another dataset, "
                f"which the model cannot read.</p>", unsafe_allow_html=True)
#4 performance
with tab_score:
    st.subheader("How well they did")
    cols = st.columns(len(results)) #card per model
    for (name, r), c in zip(results.items(), cols):
        k = r["mask"]
        acc = float((r["pred"][k] == labels[k]).mean()) #multiclass accuracy
        atk = k & (labels != "Benign")
        rec = float((r["pred"][atk] != "Benign").mean()) if atk.any() else float("nan")
        reads = "the 11 packet features" if r["reads"] is None else f"the fields of {r['reads']}"
        c.markdown(
            f"<div class='card'><div class='k'>{name}</div>"
            f"<div class='v'>{acc*100:.1f}%</div>"
            f"<div class='k' style='margin-top:10px'>correct class · {int(k.sum())} of {len(flows)} flows</div>"
            f"<div style='margin-top:12px;font-size:.85rem'>"
            f"attacks caught <b>{rec*100:.1f}%</b><br>"
            f"correct <b class='ok'>{int((r['pred'][k]==labels[k]).sum())}</b> · "
            f"wrong <b class='bad'>{int((r['pred'][k]!=labels[k]).sum())}</b><br>"
            f"reads {reads}<br>"
            f"{r['us_per_flow']:.1f} μs/flow · {int(k.sum())/max(r['seconds'],1e-9):,.0f} flows/s"
            f"</div></div>", unsafe_allow_html=True)
    #Per class recall table
    st.markdown("#### Per class")
    per = pd.DataFrame(index=sorted(set(labels)))
    per["flows"] = pd.Series(labels).value_counts()
    for name, r in results.items():
        per[name] = [float((r["pred"][sel] == c).mean()) if (sel := (labels == c) & r["mask"]).any()
                     else np.nan for c in per.index]
    st.dataframe(per.style.format({c: "{:.2f}" for c in results}, na_rep=NA), width="stretch")
    st.markdown("<p class='lead'>Share of each class's flows that was recognised correctly. "
                "Volumetric attacks are easy; the “quiet” application-layer ones are not.</p>",
                unsafe_allow_html=True)
    #Confusion matrix for the selected model
    st.markdown("#### Where it gets confused")
    pick = st.selectbox("Model", list(results))
    k = results[pick]["mask"]
    cm = pd.crosstab(pd.Series(labels[k], name="true"), pd.Series(results[pick]["pred"][k], name="predicted"))
    st.dataframe(cm, width="stretch")

with tab_one:
    #packets, featutes, model's detection for one flow
    st.subheader("One flow, the whole chain")
    f = st.selectbox("Pick a flow", list(flows.index),
                     format_func=lambda i: f"#{i} · {truth.loc[i,'label']} · {truth.loc[i,'source_dataset']} · "
                                           f"{truth.loc[i,'protocol']} · {truth.loc[i,'n_packets']} packets")
    st.markdown("**Step 1 — its packets**")
    st.dataframe(packets[packets.flow_id == f][
        ["No.", "Time", "Source", "Destination", "Protocol", "Length", "Payload", "Info"]],
        width="stretch", hide_index=True)

    st.markdown("**Step 2 — the eleven features they produce**")
    st.dataframe(flows.loc[[f]], width="stretch")
    src_ds = truth.loc[f, "source_dataset"]
    if src_ds in native and f in native[src_ds].index:
        st.markdown(f"**Step 2b — the fields of {src_ds} read by that dataset's models**")
        st.dataframe(native[src_ds].loc[[f]], width="stretch")

    st.markdown("**Step 3 — each model's decision**")
    pos = list(flows.index).index(f)  #row position of the flow in the results arrays
    truth_lab = labels[pos]
    rows = []
    for name, r in results.items():
        ok = r["mask"][pos]
        rows.append({"model": name, "predicted": r["pred"][pos],
                     "confidence": r["conf"][pos], "true": truth_lab,
                     "": ("correct" if r["pred"][pos] == truth_lab else "wrong") if ok else "not applicable"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

#footer
st.divider()
if DATA == HERE:
    #default folder
    st.markdown(
        "<p class='lead'>The models are the study's <code>combined</code> ones (<code>models_matched</code>), "
        "all three trained on ~250,000 rows (matched budget). They accept only "
        "the eleven common features and they are the only portable ones. On the full test split "
        "they flag 5.5–9.5% of benign traffic as attacks. On a real network with "
        "~0.1% attack prevalence this gives a PPV of just 0.01–0.02"
        "alerts are false (and at 1% the PPV is only 0.09–0.15).</p>", unsafe_allow_html=True)
elif any(r["reads"] for r in results.values()):
    #Alternative folder with native_<dataset>.csv
    st.markdown(
        "<p class='lead'>Models with a dataset prefix were trained only on that dataset "
        "(<code>models_matched</code>), with its Zeek fields (<code>conn_state</code>, "
        "<code>service</code>, bytes per direction, etc.). These are not derived from the demo's "
        "packets, so they are read from <code>native_&lt;dataset&gt;.csv</code> and the models are scored only "
        "on their own dataset's flows. Every flow is in its dataset's test split and "
        "in the pooled test split: no model, per-dataset or combined, has seen it.</p>", unsafe_allow_html=True)
