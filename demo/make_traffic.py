#Libraries
import sys, json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import COMMON_FEATURES, RANDOM_STATE, TEST_SIZE
from loaders import DATASETS
from sklearn.model_selection import train_test_split

PER_CLASS = 22 #flows sampled per class
MAX_PKTS = 120
CACHE = ROOT / "src" / "results" / "common_cache"
#the run that trained the shipped models
REFERENCE = ROOT / "src" / "results_matched" / "all_results.json"
rng = np.random.default_rng(RANDOM_STATE)

"""The pooled test split rebuilt as runall.run_combined + preprocessing.split"""
def held_out_flows():
    names = [n for n in DATASETS if (CACHE / f"{n}.pkl").exists()]
    if len(names) < 2:
        raise FileNotFoundError(f"need the common cache of at least two datasets in {CACHE} " f"(run src/runall.py first)")
    df = pd.concat([pd.read_pickle(CACHE / f"{n}.pkl") for n in names], ignore_index=True)
    df[COMMON_FEATURES] = (df[COMMON_FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e12, 1e12))
    df = df[~df.duplicated(subset=COMMON_FEATURES + ["label"])]
    _, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"])
    if REFERENCE.exists():
        rec = [r for r in json.loads(REFERENCE.read_text(encoding="utf-8")) if r.get("dataset") == "combined" and r.get("task") == "multiclass"]
        if rec and rec[0].get("test_rows") not in (None, len(test)):
            raise RuntimeError(f"rebuilt test split has {len(test):,} rows but the shipped models were "
                               f"scored on {rec[0]['test_rows']:,}: the cache does not match the run "
                               f"that trained them, so the sample would not be held out")
    return test

def proto_of(r):
    return "TCP" if r.proto_tcp else "UDP" if r.proto_udp else "ICMP" if r.proto_icmp else "IP"

"""Stable synthetic addresses, one pair per flow"""
def endpoints(i):
    h = int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16)
    return f"192.168.{h % 4}.{h // 4 % 254 + 1}", f"10.0.{h // 1024 % 4}.{h // 4096 % 254 + 1}"

HEADER = {"TCP": 54, "UDP": 42, "ICMP": 42, "IP": 34}   #frame overhead shown in a packet list

"""n non-negative payload sizes that sum to total (re-aggregation recovers total_bytes)"""
def sizes_for(n, total):
    if n <= 0:
        return []
    total = int(total)
    if total <= 0:
        return [0] * n
    lo = min(20, total // n)
    base = max(total // n, 1)
    s = np.full(n, base, dtype=np.int64)
    jitter = rng.integers(-base // 4, base // 4 + 1, n) if base > 8 else np.zeros(n, dtype=np.int64)
    s = np.clip(s + jitter, lo, None)
    s[-1] += total - int(s.sum())                #the last packet absorbs the difference
    if s[-1] < lo:                               #if the last packet is still too small
        deficit = int(lo - s[-1])
        s[-1] = lo
        for i in np.argsort(-s, kind="stable"):
            take = min(int(s[i]) - lo, deficit)
            s[i] -= take
            deficit -= take
            if deficit == 0:
                break
    return s.tolist()

"""First packet at t0, last at t0+duration (recover of the duration)"""
def times_for(n, t0, dur):
    if n == 1 or dur <= 0:
        return [t0] * n
    return [t0 + dur * i / (n - 1) for i in range(n)]

def info_line(proto, sp, dp, ln, first, last):
    if proto == "ICMP":
        return "Echo (ping) request" if first else "Echo (ping) reply"
    flags = "[SYN]" if first and proto == "TCP" else "[FIN, ACK]" if last and proto == "TCP" else "[ACK]"
    return f"{sp} -> {dp} {flags} Len={ln}" if proto == "TCP" else f"{sp} -> {dp} Len={ln}"

def main():
    global rng
    ap = argparse.ArgumentParser(description="rebuild demo/packets.csv and demo/flows.csv")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE,help="which held-out flows to draw (default %(default)s reproduces the shipped sample)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent, help="output folder (default: the demo folder)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    test = held_out_flows()
    usable = test[(test.total_pkts >= 1) & (test.total_pkts <= MAX_PKTS) & (test.total_bytes >= 0)]
    #one draw per class
    picked = pd.concat([g.sample(min(PER_CLASS, len(g)), random_state=args.seed)for _, g in usable.groupby("label")])
    picked = picked.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    pkts, flows, clock, no = [], [], 0.0, 1
    for fid, r in picked.iterrows():
        n     = max(int(r.total_pkts), 1)
        proto = proto_of(r)
        src, dst = endpoints(fid)
        sp, dp = int(r.src_port), int(r.dst_port)
        lens  = sizes_for(n, int(round(r.total_bytes)))
        ts    = times_for(len(lens), clock, float(r.duration))
        hdr = HEADER.get(proto, 34)
        for k, (t, ln) in enumerate(zip(ts, lens)):
            fwd = k % 2 == 0
            pkts.append({"No.": no, "Time": round(t, 9),
                         "Source": src if fwd else dst, "Destination": dst if fwd else src,
                         "Protocol": proto, "Length": int(ln) + hdr, "Payload": int(ln),
                         "SrcPort": sp if fwd else dp, "DstPort": dp if fwd else sp,
                         "Info": info_line(proto, sp if fwd else dp, dp if fwd else sp, int(ln),k == 0, k == len(lens) - 1),
                         "flow_id": fid})
            no += 1
        flows.append({"flow_id": fid, "src": src, "dst": dst, "protocol": proto,"src_port": sp, "dst_port": dp, "n_packets": len(lens),
                      "label": r.label, "source_dataset": r.source})
        clock = ts[-1] + float(rng.uniform(0.002, 0.25))   #gap before the next conversation

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pkts).to_csv(args.out / "packets.csv", index=False)
    pd.DataFrame(flows).to_csv(args.out / "flows.csv", index=False)
    print(f"packets.csv: {len(pkts):,} packets")
    print(f"flows.csv  : {len(flows):,} flows over {len(picked.label.unique())} classes")
    print(picked.label.value_counts().to_string())

if __name__ == "__main__":
    main()
