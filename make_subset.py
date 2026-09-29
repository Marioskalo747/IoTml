#Libraries
import sys, json, shutil, argparse
from pathlib import Path
import numpy as np, pandas as pd, joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

import preprocessing
preprocessing.record_split_audit = lambda *a, **k: None   #never touch the study's split_audit.json
from preprocessing import build_xy, split
from config import COMMON_FEATURES, RANDOM_STATE
from loaders import DATASETS
import make_traffic as mt
from flowize import packets_to_flows, FEATURES, read_native, native_input

MODELS    = {"ton_iot": ["XGBoost", "CatBoost"], "iot23": ["XGBoost", "CatBoost"]}
PER_CLASS = 20
SRC_MODELS = ROOT / "src" / "models_matched"
RECORDED  = ROOT / "src" / "results_matched" / "all_results.json"
    
"""Row positions, per source dataset, of the pooled test split of the combined models"""
def pooled_test_rows():
    names = [n for n in DATASETS if (mt.CACHE / f"{n}.pkl").exists()]
    sizes = [len(pd.read_pickle(mt.CACHE / f"{n}.pkl")) for n in names]
    offset = dict(zip(names, np.cumsum([0] + sizes[:-1]))) #start offset of each dataset
    test = mt.held_out_flows()          #checks the test size against the shipped run
    return {n: pd.Index(test.index[test["source"] == n] - offset[n]) for n in names}


"""The dataset's own split"""
def rebuild(ds, recorded):
    loader, to_common = DATASETS[ds]
    df = loader() #raw dataset with its zeek fields
    common = to_common(df) #common features
    cached = pd.read_pickle(mt.CACHE / f"{ds}.pkl")
    if not (len(cached) == len(common)
            and np.allclose(common[COMMON_FEATURES].to_numpy(float), cached[COMMON_FEATURES].to_numpy(float))
            and (common["label"].values == cached["label"].values).all()):
        raise RuntimeError(f"{ds}: the loader no longer reproduces common_cache, so the pooled "
                           f"test rows cannot be mapped back to it")
    #Per-dataset split identical to training
    X, y, yb = build_xy(df)
    del df
    _, X_te, _, y_te, _, _ = split(X, y, yb, dataset=ds)
    models = {}
    for name in MODELS[ds]:
        m = joblib.load(SRC_MODELS / f"{ds}__multiclass__{name}.joblib")
        rec = recorded[(ds, name)]
        acc = float((np.asarray(m.predict(X_te)).ravel().astype(str) == y_te.values).mean())
        if len(X_te) != rec["test_rows"] or abs(acc - rec["accuracy"]) > 1e-9:
            raise RuntimeError(f"{ds} {name}: rebuilt split gives {len(X_te):,} rows / accuracy {acc:.6f}, "
                               f"the model was scored on {rec['test_rows']:,} / {rec['accuracy']:.6f}")
        print(f"{ds} {name}: split verified, {len(X_te):,} test rows, accuracy {acc:.6f}")
        models[name] = m
    return common, X_te, y_te.astype(str), models

def main():
    ap = argparse.ArgumentParser(description="rebuild the TON-IoT + IoT-23 demo capture")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = ap.parse_args()
    mt.rng = np.random.default_rng(args.seed)        #sizes_for draws from make_traffic's rng

    recorded = {(r["dataset"], r["model"]): r for r in json.loads(RECORDED.read_text(encoding="utf-8")) 
                if r.get("task") == "multiclass" and not r.get("tuned") and r.get("dataset") in MODELS}
    pooled = pooled_test_rows()

    picked, native, models = [], {}, {}
    for ds in MODELS:
        common, X_te, y_te, models[ds] = rebuild(ds, recorded)
        held = X_te.index.intersection(pooled[ds])
        c = common.loc[held].assign(label=y_te.loc[held].values, source=ds, row=held)
        usable = c[(c.total_pkts >= 1) & (c.total_pkts <= mt.MAX_PKTS) & (c.total_bytes >= 0)]
        #share of each class in the held-out flows that is within the packet-cap (some classes may be entirely excluded)
        cover = (usable.label.value_counts() / c.label.value_counts()).fillna(0)
        print(f"{ds}: {len(held):,} rows held out from both splits; packet-cap coverage per class " + ", ".join(f"{k} {v:.1%}" for k, v in cover.sort_index().items()))
        picked += [g.sample(min(PER_CLASS, len(g)), random_state=args.seed) for _, g in usable.groupby("label")]
        native[ds] = X_te
    #Merge both datasets and shuffle
    picked = pd.concat(picked).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    #packet synthesis
    pkts, flows, clock, no = [], [], 0.0, 1
    for fid, r in picked.iterrows():
        n = max(int(r.total_pkts), 1)
        proto = mt.proto_of(r)
        src, dst = mt.endpoints(fid)
        sp, dp = int(r.src_port), int(r.dst_port)
        lens = mt.sizes_for(n, int(round(r.total_bytes))) #payloads summing
        ts = mt.times_for(len(lens), clock, float(r.duration)) #timestamps spanning the duration 
        hdr = mt.HEADER.get(proto, 34)
        for k, (t, ln) in enumerate(zip(ts, lens)): 
            fwd = k % 2 == 0 #alternate forward/backward packets
            pkts.append({"No.": no, "Time": round(t, 9),
                         "Source": src if fwd else dst, "Destination": dst if fwd else src,
                         "Protocol": proto, "Length": int(ln) + hdr, "Payload": int(ln),
                         "SrcPort": sp if fwd else dp, "DstPort": dp if fwd else sp,
                         "Info": mt.info_line(proto, sp if fwd else dp, dp if fwd else sp, int(ln), k == 0, k == len(lens) - 1),
                         "flow_id": fid})
            no += 1
        flows.append({"flow_id": fid, "src": src, "dst": dst, "protocol": proto,
                      "src_port": sp, "dst_port": dp, "n_packets": len(lens),
                      "label": r.label, "source_dataset": r.source})
        clock = ts[-1] + float(mt.rng.uniform(0.002, 0.25)) #idle gap between flows
    #Write the output 
    pd.DataFrame(pkts).to_csv(HERE / "packets.csv", index=False)
    pd.DataFrame(flows).to_csv(HERE / "flows.csv", index=False)
    for ds, X_te in native.items():
        rows = picked[picked.source == ds]
        tbl = X_te.loc[rows.row.values].copy() #native fields of the sampled flows
        tbl.insert(0, "flow_id", rows.index.values)
        tbl.to_csv(HERE / f"native_{ds}.csv", index=False)
    (HERE / "models").mkdir(exist_ok=True)
    for ds, names in MODELS.items():
        for name in names:
            shutil.copy2(SRC_MODELS / f"{ds}__multiclass__{name}.joblib", HERE / "models" / f"{ds}__{name}.joblib")
    print(f"\npackets.csv: {len(pkts):,} packets   flows.csv: {len(flows):,} flows")
    print(picked.groupby(["source", "label"]).size().to_string())

    #checks on what was written, read back the way the demo reads it
    truth = pd.read_csv(HERE / "flows.csv").set_index("flow_id")
    fl = packets_to_flows(pd.read_csv(HERE / "packets.csv"))
    same = {f: int(np.isclose(fl[f].values, picked.loc[fl.index, f].values, rtol=1e-6, atol=1e-6).sum()) for f in FEATURES}
    print("\nfeatures recovered from the packets (of %d): " % len(fl) + ", ".join(f"{k} {v}" for k, v in same.items()))
    for ds, names in MODELS.items():
        tbl = read_native(HERE / f"native_{ds}.csv")
        lab = truth.loc[tbl.index, "label"].astype(str).values
        ref = native[ds].loc[picked.loc[tbl.index, "row"].values]
        for name in names:
            m = models[ds][name]
            p_csv = np.asarray(m.predict(native_input(m, tbl))).ravel().astype(str)
            p_mem = np.asarray(m.predict(ref)).ravel().astype(str)
            print(f"{ds}__{name}: CSV round trip identical {int((p_csv == p_mem).sum())}/{len(p_csv)}, " 
                  f"correct class {(p_csv == lab).mean():.1%} on its {len(lab)} flows")

if __name__ == "__main__":
    main()
