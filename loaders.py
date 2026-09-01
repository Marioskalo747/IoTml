#Libraries
import io 
import sys
import tarfile 
import zipfile 
from pathlib import Path
import numpy as np
import pandas as pd
import shutil #external drive copy
import time as _t
import json as _json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (MAX_ROWS_PER_DATASET, MIN_CLASS_ROWS, MIN_ROWS_PER_CLASS_LOAD, RANDOM_STATE,TS_COL, dataset_dir)  

RNG = np.random.RandomState(RANDOM_STATE)

'''File search across glob patterns'''
def find(root: Path, pattern: list[str]):
    out: list[Path] = []
    if not root.exists():
        return out
    for p in pattern:
        out.extend(root.rglob(p))
    return sorted({p for p in out if "_local_cache" not in str(p) }) #set removes duplicates

'''Preserve capture time in the reserved TS_COL so split() order rows chronologically'''
def attach_timestamp(df: pd.DataFrame, candidates):
    for c in candidates:
        if c not in df.columns:
            continue
        ts = pd.to_numeric(df[c], errors="coerce")
        if ts.notna().mean() < 0.5: #not epoch seconds, try to parse it as a date string
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            if parsed.notna().mean() < 0.5:
                continue #this column is not a usable timestamp
            ts = parsed.astype("int64") / 1e9
        if ts.notna().mean() >= 0.5 and ts.nunique(dropna=True) > 1: #a constant clock is useless
            df[TS_COL] = ts.astype("float64")
            return df
    return df

'''Stratified row cap (reduction per class)'''
def stratified_cap(df: pd.DataFrame, cap: int, label_col: str = "label"):
    if len(df) <= cap:
        return df.reset_index(drop=True)
    frac = cap/len(df)
    parts = []
    for _, grp in df.groupby(label_col, observed = True):
        #min floor stops rare attacks from vanishing, shuffle breaks per class ordering
        n= max(int(round(len(grp)*frac)), min(len(grp), MIN_ROWS_PER_CLASS_LOAD))
        parts.append(grp.sample(n=min(n, len(grp)), random_state=RANDOM_STATE))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

'''Merge rare classes into "Other"'''
def merge_rare_classes(df: pd.DataFrame, label_col: str = "label"):
    counts = df[label_col].value_counts()
    rare = [c for c in counts[counts < MIN_CLASS_ROWS].index if c != "Benign"]
    if len(rare):
        df = df.copy()
        df.loc[df[label_col].isin(rare), label_col]= "Other"
        #small Other class dropped 
        if (df[label_col] == "Other").sum() < MIN_CLASS_ROWS:  #########  ELEGXOS
            df = df[df[label_col] != "Other"]
    return df.reset_index(drop=True)

'''One hot encoding of protocol column(both names and IANA numbers)'''
def proto_onehot(series: pd.Series):
    s = series.astype(str).str.lower()
    return pd.DataFrame({"proto_tcp": (s.str.contains("tcp") | (s == "6")).astype(np.int8),
                         "proto_udp": (s.str.contains("udp") | (s == "17")).astype(np.int8), 
                         "proto_icmp": (s.str.contains("icmp") | (s == "1")).astype(np.int8)})
    #different protocol writing in datasets    
    
'''Save division with 0/0 and inf/inf'''   
def safe_div(a,b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    safe_b = np.where(b > 0, b, 1) #temporary denominator avoids RuntimeWarning
    division = a / safe_b
    condition = (b > 0) & np.isfinite(b) #quotient is valid only if denominator
    out = np.where(condition, division, 0.0)
    idx = getattr(a, "index", None) #preserve index
    if idx is None:
        idx = getattr(b, "index", None)
    return pd.Series(out, index=idx)
    
'''Port numbers that may be written in hex or decimal'''
def parse_port(series: pd.Series):
    s = series.astype(str).str.strip().str.lower()
    is_hex = s.str.startswith("0x")
    out = pd.to_numeric(s.mask(is_hex), errors="coerce") #decimal branch
    if is_hex.any():
        #a truncated or corrupt field ("0x", "0xZZ") makes int(v, 16) raise
        def _hex(v):
            try:
                return int(v, 16)
            except (TypeError, ValueError):
                return np.nan
        hx = s.where(is_hex).map(_hex)
        out = out.fillna(hx)
    out = out.fillna(-1)
    return out.where(out.between(-1, 65535), -1) #a port can never exceed 65535


WINDOWS = (5.0, 10.0) #seconds of network activity summarised around each flow

'''Windowed context features (what else was happening on the wire around this flow)'''
def add_window_features(df: pd.DataFrame, ts_col, src_col, dst_col, dport_col=None,
                        bytes_col=None, windows=WINDOWS):
    need = [c for c in (ts_col, src_col, dst_col) if c not in df.columns]
    if need:
        return df, []
    ts = pd.to_numeric(df[ts_col], errors="coerce")
    if ts.notna().mean() < 0.5 or ts.nunique(dropna=True) < 2:
        return df, [] #no usable clock, nothing to window over
    t = ts.fillna(ts.min()).to_numpy(dtype="float64")
    src = pd.factorize(df[src_col].astype(str).to_numpy())[0]
    dst = pd.factorize(df[dst_col].astype(str).to_numpy())[0]
    dpt = (pd.factorize(df[dport_col].astype(str).to_numpy())[0]
           if dport_col in df.columns else src)
    byt = (pd.to_numeric(df[bytes_col], errors="coerce").fillna(0).to_numpy(dtype="float64")
           if bytes_col in df.columns else np.zeros(len(t)))
    made = []
    for W in windows:
        tag = f"w{int(W)}"
        f_dst, b_dst, s_dst = _causal_window(t, dst, W, values=byt, distinct_of=src)
        f_src, _, d_src = _causal_window(t, src, W, distinct_of=dst)
        _, _, p_src = _causal_window(t, src, W, distinct_of=dpt)
        for name, arr in ((f"{tag}_flows_to_dst", f_dst), (f"{tag}_srcs_to_dst", s_dst),
                          (f"{tag}_bytes_to_dst", b_dst), (f"{tag}_flows_from_src", f_src),
                          (f"{tag}_dsts_from_src", d_src), (f"{tag}_dports_from_src", p_src)):
            df[name] = arr
            made.append(name)
    return df, made

'''Counts, sums and DISTINCT counts over the causal window [t-W, t], per group key'''
def _causal_window(t, codes, W, values=None, distinct_of=None):
    n = len(t)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    #lexsort puts each group contiguous and in time order (every group is a plain numpy slice)
    order = np.lexsort((t, codes))
    cs, ts_ = codes[order], t[order]
    starts = np.flatnonzero(np.concatenate(([True], cs[1:] != cs[:-1])))
    ends = np.concatenate((starts[1:], [n]))
    vs = values[order] if values is not None else None
    ds = distinct_of[order] if distinct_of is not None else None
    cnt = np.zeros(n); tot = np.zeros(n); dis = np.zeros(n)
    for a, e in zip(starts, ends):
        tg = ts_[a:e]
        left = np.searchsorted(tg, tg - W, side="left")
        pos = np.arange(e - a)
        cnt[a:e] = pos - left + 1
        if vs is not None:
            c = np.concatenate(([0.0], np.cumsum(vs[a:e])))
            tot[a:e] = c[pos + 1] - c[left]
        if ds is not None:
            dv = ds[a:e]
            counter: dict = {}
            lo = 0; nd = 0
            out = np.empty(e - a)
            for hi in range(e - a):
                v = dv[hi]
                c0 = counter.get(v, 0)
                if c0 == 0:
                    nd += 1
                counter[v] = c0 + 1
                while tg[lo] < tg[hi] - W:
                    u = dv[lo]
                    counter[u] -= 1
                    if counter[u] == 0:
                        nd -= 1
                    lo += 1
                out[hi] = nd
            dis[a:e] = out
    #scatter back to the caller's row order
    o_cnt = np.empty(n); o_tot = np.empty(n); o_dis = np.empty(n)
    o_cnt[order] = cnt; o_tot[order] = tot; o_dis[order] = dis
    return o_cnt, o_tot, o_dis

'''Shared final step of every converter'''
def finalize_common(common: pd.DataFrame, df: pd.DataFrame):
    common["label"] = df["label"].values #ignore any index misalignment
    common["binary"] = df["binary"].values
    common = common.replace([np.inf, -np.inf], np.nan).fillna(0)
    return common 

'''Copy dataset locally if on external drive'''
def localize(path: Path):
    if path.drive.lower() == Path(__file__).drive.lower():
        return path #same drive
    cache = Path(__file__).resolve().parents[1] / "datasets/_local_cache"   ####### 1 h 0
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / path.name
    size = path.stat().st_size
    if dest.exists() and dest.stat().st_size == size:
        return dest #already copied
    free = shutil.disk_usage(cache).free
    if free < size* 1.2: #20% margin for temp files
        raise OSError(f"not enough free space in {cache} to copy {path}, only {free/1e9:.1f} GB free)")
    last_err: Exception | None = None
    for attempt in range(1,4): #3 attempts with 30s wait
        try:
            print(f"copying {path.name} to local disk {size/1e9:.1f}GB, Try:{attempt}/3...")
            shutil.copy2(path, dest)
            if dest.stat().st_size == size:
                return dest
            raise OSError(f"size mismatch")
        except OSError as e:
            last_err = e
            dest.unlink(missing_ok=True) #delete partial copy
            _t.sleep(30)
    raise OSError(f"Failed to localze {path}: {last_err}")



#Detection patterns
BUILTIN_PROBES = {"bot_iot": ["bot_iot*.zip", "5%*.zip", "*5pc*.zip", "*bot*iot*.zip", "*Full5pc*.csv"],
    "ton_iot": ["ton_iot*.csv", "Train_Test_Network*.csv", "train_test_network*.csv"],
    "ciciot2023": ["ciciot2023*.zip", "CSV.zip", "*CICIoT*.zip", "*ciciot*.zip", "*.pcap.csv"],
    "iot23": ["iot_23*.tar.gz", "iot_23*small*.tar.gz", "*iot*23*.tar.gz", "conn.log.labeled", "*.log.labeled"]}
    
DATASET_CONFIG_PATH = Path(__file__).resolve().parents[1] / "datasets_config.json"

BUILTIN_DATASETS = {"bot_iot", "ton_iot", "ciciot2023", "iot23"}

#label column names in priority order
LABEL_COL_CANDIDATES = ["label", "attack", "attack_type", "attack_cat", "category", "type","class", "marker", "target", "traffic_type"]
GENERIC_DROP = ["flow_id", "src_ip", "dst_ip", "source_ip", "destination_ip", "src_host","dst_host", "ip.src_host", "ip.dst_host", "timestamp", "ts", "frame.time",
    "time", "uid", "id", "no", "no.", "flow_start", "flow_end"]

#column name synonyms
DUR = ["duration", "flow_duration", "dur", "flow_dur"]
PKTS = ["total_pkts", "tot_pkts", "pkts", "packets", "total_packets", "number"]
BYTES = ["total_bytes", "tot_bytes", "bytes", "total_size", "tot_size", "tot_sum"]
DPORT = ["dst_port", "dport", "destination_port", "id.resp_p", "tcp.dstport"]
SPORT = ["src_port", "sport", "source_port", "id.orig_p", "tcp.srcport"]
PROTO = ["proto", "protocol", "protocol_type", "ip.proto"]
FWD_PKTS = ["fwd_pkts", "tot_fwd_pkts", "total_fwd_packets", "src_pkts", "orig_pkts"]
BWD_PKTS = ["bwd_pkts", "tot_bwd_pkts", "total_backward_packets", "dst_pkts", "resp_pkts"]
FWD_BYTES = ["fwd_bytes", "totlen_fwd_pkts", "src_bytes", "orig_bytes", "orig_ip_bytes"]
BWD_BYTES = ["bwd_bytes", "totlen_bwd_pkts", "dst_bytes", "resp_bytes", "resp_ip_bytes"]

#Each dataset's taxonomy
BOTIOT_LABEL_MAP = {"normal": "Benign", "ddos": "DDoS", "dos": "DoS", "reconnaissance": "Recon", "theft": "Theft"}
BOTIOT_DROP = ["pkSeqID", "stime", "ltime", "seq", "saddr", "daddr", "attack", "category", "subcategory"] #identities and label leaks
TONIOT_LABEL_MAP = {"normal": "Benign", "ddos": "DDoS", "dos": "DoS", "scanning": "Recon", "password": "BruteForce",
    "xss": "Web", "injection": "Injection", "backdoor": "Botnet", "ransomware": "Ransomware", "mitm": "Spoofing_MITM"}
TONIOT_DROP = ["ts", "src_ip", "dst_ip", "dns_query", "ssl_subject", "ssl_issuer","http_uri", "http_referrer", "http_user_agent", "weird_name","weird_addl", "weird_notice", 
               "http_orig_mime_types","http_resp_mime_types", "ssl_cipher", "ssl_version", "http_version","http_method", "http_trans_depth", "dns_rcode", "dns_qclass", "dns_qtype"]
#Zeek conn.log.labeled columns
IOT23_COLS = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service", "duration", 
    "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
    "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents", "label", "detailed-label"]

'''Reads BoT-IoT CSV with low_memory=False'''
def read_botiot_csv(fh):
    return pd.read_csv(fh, low_memory=False)

'''BoT-IoT dataset loader (zip and csv)'''
def load_bot_iot(cap: int = MAX_ROWS_PER_DATASET):
    root = dataset_dir()
    parts = []
    zips = find(root, ["bot_iot*.zip", "5%*.zip", "*5pc*.zip", "*bot*iot*.zip"])       ###ONOMATA
    csvs = [f for f in find(root, ["*Full5pc*.csv"]) if f.suffix == ".csv"]
    if zips:
        with zipfile.ZipFile(localize(zips[0])) as zf:
            #5% subset
            members = [m for m in zf.namelist() if "Full5pc" in m and m.endswith(".csv")]   #######ONOMA
            for m in members:
                with zf.open(m) as fh:
                    #TextIOWrapper zipfile yields bytes, pandas expects text
                    parts.append(read_botiot_csv(io.TextIOWrapper(fh, errors="ignore")))
    elif csvs:
        parts = [read_botiot_csv(f) for f in csvs]
    if not parts:
        raise FileNotFoundError(f"no BoT-IoT CSV found in {root}")
    df = pd.concat(parts, ignore_index=True)
    df["label"] = (df["category"].astype(str).str.strip().str.lower().map(BOTIOT_LABEL_MAP).fillna("Other"))
    df["binary"] = (df["label"] != "Benign").astype(np.int8)
    df = attach_timestamp(df, ["stime", "ltime"]) #keep capture time before BOTIOT_DROP removes it
    #category, subcatgory and attack are dropped (label leak)
    df, _ = add_window_features(df, "stime", "saddr", "daddr", dport_col="dport", bytes_col="bytes")
    df = df.drop(columns=[c for c in BOTIOT_DROP if c in df.columns])
    for c in ["sport", "dport"]:
        #ports in hexadecimal to decimal conversion
        df[c] = parse_port(df[c])
    df = merge_rare_classes(df, label_col="label")
    return stratified_cap(df, cap)

'''BoT-IoT common feature extraction'''
def bot_iot_common(df: pd.DataFrame):
    common = proto_onehot(df["proto"])
    common["duration"] = pd.to_numeric(df["dur"], errors="coerce")
    common["total_pkts"] = pd.to_numeric(df["pkts"], errors="coerce")
    common["total_bytes"] = pd.to_numeric(df["bytes"], errors="coerce")
    common["pkt_rate"] = safe_div(common["total_pkts"], common["duration"])
    common["byte_rate"] = safe_div(common["total_bytes"], common["duration"])
    common["avg_pkt_size"] = safe_div(common["total_bytes"], common["total_pkts"])
    common["dst_port"] = pd.to_numeric(df["dport"], errors="coerce").fillna(0)
    common["src_port"] = pd.to_numeric(df["sport"], errors="coerce").fillna(0)
    return finalize_common(common, df)

'''TON-IoT dataset loader'''
def load_ton_iot(cap: int = MAX_ROWS_PER_DATASET):
    root = dataset_dir()
    files = find(root, ["ton_iot*.csv", "Train_Test_Network*.csv", "train_test_network*.csv"])   #####ONOMATA
    if not files:
        raise FileNotFoundError(f"no TON-IoT CSV found in {root}")
    df = pd.read_csv(files[0], low_memory = False)
    df.columns = [c.strip().lower() for c in df.columns] #normalize column names
    df["label_raw"] = df["type"].astype(str).str.strip().str.lower()
    df["label"] = df["label_raw"].map(TONIOT_LABEL_MAP).fillna("Other")
    df["binary"] = (df["label"] != "Benign").astype(np.int8)
    df = attach_timestamp(df, ["ts"]) #keep capture time before TONIOT_DROP removes it
    #drop identities and high-cardinality columns (label leak)
    df = df.drop(columns=[c for c in TONIOT_DROP + ["type", "label_raw"] if c in df.columns])
    df = df.replace("-", np.nan) #zeek style missing value
    df = merge_rare_classes(df)
    return stratified_cap(df, cap)

'''TON-IoT common feature extraction'''
def ton_iot_common(df: pd.DataFrame):
    common = proto_onehot(df["proto"])
    common["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    sp = pd.to_numeric(df.get("src_pkts"), errors="coerce").fillna(0)
    dp = pd.to_numeric(df.get("dst_pkts"), errors="coerce").fillna(0)
    sb = pd.to_numeric(df.get("src_bytes"), errors="coerce").fillna(0)
    db = pd.to_numeric(df.get("dst_bytes"), errors="coerce").fillna(0)
    common["total_pkts"] = sp + dp
    common["total_bytes"] = sb + db
    common["pkt_rate"] = safe_div(common["total_pkts"], common["duration"])
    common["byte_rate"] = safe_div(common["total_bytes"], common["duration"])   
    common["avg_pkt_size"] = safe_div(common["total_bytes"], common["total_pkts"])
    common["dst_port"] = pd.to_numeric(df.get("dst_port"), errors="coerce").fillna(0)
    common["src_port"] = pd.to_numeric(df.get("src_port"), errors="coerce").fillna(0)
    return finalize_common(common, df)

'''CICIOT2023 dataset loader unifying taxonomy'''
def ciciot_map_label(raw: str):
    s = str(raw).strip().lower()
    if "benign" in s:
        return "Benign"
    if "mirai" in s:
        return "Mirai"
    if "ddos" in s:
        return "DDoS" #order matters(ddos test before dos)
    if "dos" in s:
        return "DoS"
    if any(k in s for k in ("recon", "vulnerabilityscan", "scan", "ping_sweep", "os_scan", "port_scan", "hostdiscovery")):
        return "Recon"
    if any(k in s for k in ("bruteforce", "dictionary")):
        return "BruteForce"
    #Injection and Botnet are checked BEFORE Web on purpose
    if any(k in s for k in ("sqlinjection", "commandinjection", "sql", "command", "injection")):
        return "Injection"
    if "backdoor" in s:
        return "Botnet"
    if any(k in s for k in ("web", "xss", "uploading", "browserhijacking")):
        return "Web"
    if any(k in s for k in ("spoofing", "mitm")):
        return "Spoofing_MITM"
    return "Other"

'''Chunked sampling of CICIOT2023 CSVs to avoid memory issues'''
def ciciot_read_sampled(fh, per_file: int):
    chunks = []
    got = 0
    for chunk in pd.read_csv(fh, low_memory=False, chunksize=250000):
        take = min(per_file, max(per_file//4, 2000), len(chunk))
        chunks.append(chunk.sample(n=take, random_state=RANDOM_STATE) if len(chunk) > take else chunk)
        got += take
        if got >= per_file*2:
            break #early stop if enough rows sampled
    df= pd.concat(chunks, ignore_index=True)
    if len(df) > per_file:
        df = df.sample(n=per_file, random_state=RANDOM_STATE) #second sampling 
    return df 
    
'''Row budget per CSV, allocated per class instead of per file'''
def ciciot_budget(members, name_of, cap: int):
    by_label: dict[str, list] = {}
    for m in members:
        by_label.setdefault(ciciot_map_label(name_of(m)), []).append(m)
    per_class = max(MIN_ROWS_PER_CLASS_LOAD, (cap * 2) // max(len(by_label), 1))
    out = {}
    for label, files in by_label.items():
        share = max(per_class // max(len(files), 1), 1000) #never read less than 1000 from a file
        for m in files:
            out[m] = share
    return out

'''CICIOT2023 CSV per class inside zip'''
def load_ciciot2023(cap: int = MAX_ROWS_PER_DATASET):
    root = dataset_dir()
    parts: list[pd.DataFrame] = []
    zips = find(root, ["ciciot2023*.zip", "CSV.zip", "*CICIoT*.zip", "*ciciot*.zip"])   ##### ONOMATA
    csvs = [f for f in find(root, ["*.pcap.csv"])]
    if zips:
        with zipfile.ZipFile(localize(zips[0])) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError(f"no CSV found in {zips[0]}")
            budget = ciciot_budget(members, lambda m: Path(m).parent.name or Path(m).stem, cap)
            for m in members:
                raw = Path(m).parent.name or Path(m).stem #folder carries attack name
                with zf.open(m) as fh:
                    df = ciciot_read_sampled(io.TextIOWrapper(fh, errors="ignore"), budget[m])
                    df["temp_label"] = raw
                    parts.append(df)
    elif csvs:
        budget = ciciot_budget(csvs, lambda f: f.parent.name, cap)
        for f in csvs: 
            df = ciciot_read_sampled(f, budget[f])
            df["temp_label"] = f.parent.name
            parts.append(df)
    else:
        raise FileNotFoundError(f"no CICIoT2023 CSV found in {root}")
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "label" in df.columns:
        df["label"] = df["label"].map(ciciot_map_label) #sone releases carry label column
    else:
        df["label"] = df["temp_label"].map(ciciot_map_label) #others only the filename
    df = df.drop(columns=["temp_label"], errors="ignore")
    df["binary"] = (df["label"] != "Benign").astype(np.int8)
    df = merge_rare_classes(df)
    return stratified_cap(df, cap)

'''CICIOT2023 common feature extraction'''
def ciciot_common(df: pd.DataFrame):
    #protocols already binary columns (unnecessary one-hot)
    common = pd.DataFrame({
        "proto_tcp": pd.to_numeric(df.get("tcp"), errors="coerce").fillna(0).astype(np.int8),
        "proto_udp": pd.to_numeric(df.get("udp"), errors="coerce").fillna(0).astype(np.int8),
        "proto_icmp": pd.to_numeric(df.get("icmp"), errors="coerce").fillna(0).astype(np.int8),
    })
    common["total_pkts"] = pd.to_numeric(df.get("number"), errors="coerce")
    common["avg_pkt_size"] = pd.to_numeric(df.get("avg"), errors="coerce")
    if "flow_duration" in df.columns:
        common["duration"] = pd.to_numeric(df["flow_duration"], errors="coerce")
    else:
        #duration is missing in some releases, use inter-arrival time and total packets to estimate
        common["duration"] = (pd.to_numeric(df.get("iat"), errors="coerce").abs()*common["total_pkts"])
    #triple fallback for total bytes, some releases carry tot_sum, others tot_size, others neither
    common["total_bytes"] = pd.to_numeric(df.get("tot_sum"), errors="coerce").fillna(pd.to_numeric(df.get("tot_size"), errors="coerce")).fillna(common["total_pkts"]*common["avg_pkt_size"])
    common["pkt_rate"] = pd.to_numeric(df.get("rate"), errors = "coerce").fillna(safe_div(common["total_pkts"], common["duration"]))
    common["byte_rate"] = safe_div(common["total_bytes"], common["duration"])
    common["dst_port"] = 0 #no ports
    common["src_port"] = 0
    return finalize_common(common, df)

'''IoT23 combines coarse label with detailed label'''
def iot23_map_label(label: str, detail: str):
    l, d = str(label).strip().lower(), str(detail).strip().lower()
    if "benign" in l:
        return "Benign"
    if "ddos" in d or "ddos" in l:
        return "DDoS"
    if "portscan" in d or "scan" in d:
        return "Recon"
    if "mirai" in d:
        return "Mirai"
    #Botnet families
    if any(k in d for k in ("c&c", "cc", "heartbeat", "torii", "okiru", "gafgyt", "hakai", "muhstik", "hide")):
        return "Botnet"
    if "filedownload" in d or "attack" in d:
        return "Botnet"
    return "Other" if l else "Benign" #empty label is treated as benign

'''Chunked read of Zeek (fallback for irregular field separation)'''
def parse_zeek_stream(fh, name: str, per_file: int):
    chunks = []
    got = 0
    try:
        for chunk in pd.read_csv(fh, sep="\t", comment="#", header=None, names=IOT23_COLS, na_values=["-", "(empty)"], on_bad_lines="skip", chunksize=400000, dtype=str):
            take = min(per_file//3+1, len(chunk))
            #contiguous blocks, not a random sample
            chunks.append(_windowed_blocks(chunk, take))
            got+=len(chunks[-1])
            if got>=per_file*2:
                break
    except Exception:
        pass #currupt log keep what read instead of losing the whole dataset
    if not chunks:
        return None
    df = pd.concat(chunks, ignore_index=True)
    #fallback 3 last fields are sometimes seperated by space instead of tab
    if df["label"].isna().mean() >0.5:
        parts3 = df["tunnel_parents"].astype(str).str.split(expand=True)
        if parts3.shape[1] >= 3:
            df["tunnel_parents"]= parts3[0]
            df["label"] = parts3[1]
            df["detailed-label"] = parts3[2]
    if len(df) > per_file:
        df = df.sample(n=per_file, random_state=RANDOM_STATE)
    df["capture"] = name #which scenario (auditing)
    return df

'''Contiguous blocks of a chunk, each carrying its own windowed context'''
def _windowed_blocks(chunk: pd.DataFrame, take: int, n_blocks: int = 4, warmup: int = 3000):
    if len(chunk) <= take:
        out, _ = add_window_features(chunk, "ts", "id.orig_h", "id.resp_h",
                                     dport_col="id.resp_p", bytes_col="orig_ip_bytes")
        return out
    size = max(take // n_blocks, 1)
    starts = np.linspace(0, len(chunk) - size, n_blocks).astype(int)
    out = []
    for st in starts:
        lo = max(0, int(st) - warmup)
        block = chunk.iloc[lo:int(st) + size].copy()
        block, _ = add_window_features(block, "ts", "id.orig_h", "id.resp_h",
                                       dport_col="id.resp_p", bytes_col="orig_ip_bytes")
        out.append(block.iloc[int(st) - lo:]) #drop the warm-up rows, keep their effect
    return pd.concat(out, ignore_index=True)

'''IoT23 dataset loader (tar.gz and conn.log.labeled)'''
def load_iot23(cap: int = MAX_ROWS_PER_DATASET):
    root = dataset_dir()
    parts: list[pd.DataFrame] = []
    tars = find(root, ["iot_23*.tar.gz", "iot_23*small*.tar.gz", "*iot*23*.tar.gz"])   #####ONOMATA
    logs = find(root, ["conn.log.labeled", "*.log.labeled"])                          #####ONOMATA
    per_file = 60000
    if logs:
        for f in logs:
            with open(f, "r", errors="ignore") as fh:
                df = parse_zeek_stream(fh, f.parent.name, per_file)
            if df is not None:
                parts.append(df)
    elif tars:
        local = localize(tars[0])
        with tarfile.open(local, "r:gz") as tf:
            for member in tf: #tar does not extract to disk
                if not member.name.endswith("conn.log.labeled"):
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                #capture name is the first subfolder
                name = Path(member.name).parts[1] if len(Path(member.name).parts) > 1 else Path(member.name).parts[0]
                df = parse_zeek_stream(io.TextIOWrapper(fh, errors="ignore"), name, per_file)
                if df is not None:
                    parts.append(df)
                    print(f" iot23 {name}: {len(df):,} rows sampled")
    if not parts:
        raise FileNotFoundError(f"no IoT23 conn.log.labeled found in {root}")
    df = pd.concat(parts, ignore_index=True)
    df["label"]=[ iot23_map_label(l, d) for l, d in zip(df["label"], df["detailed-label"])]
    df["binary"]= (df["label"] != "Benign").astype(np.int8)
    df = attach_timestamp(df, ["ts"]) #Zeek epoch time, kept for the temporal split only
    #only behavioral features are kept (no identities, no timestamps, no history)
    keep=["id.orig_p", "id.resp_p", "proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
          "missed_bytes", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "label", "binary",]
    #the windowed context columns are behavioural not identities
    keep += [c for c in df.columns if c.startswith(("w5_", "w10_"))]
    df = df[[c for c in keep + [TS_COL] if c in df.columns]] #TS_COL is not a feature, split() drops it
    df = merge_rare_classes(df)
    return stratified_cap(df, cap)

'''IoT23 common feature extraction'''
def iot23_common(df: pd.DataFrame):
    common = proto_onehot(df["proto"])
    common["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    op = pd.to_numeric(df["orig_pkts"], errors="coerce").fillna(0)
    rp = pd.to_numeric(df["resp_pkts"], errors="coerce").fillna(0)
    ob = pd.to_numeric(df["orig_ip_bytes"], errors="coerce").fillna(0)
    rb = pd.to_numeric(df["resp_ip_bytes"], errors="coerce").fillna(0)
    common["total_pkts"] = op + rp
    common["total_bytes"] = ob + rb
    common["pkt_rate"] = safe_div(common["total_pkts"], common["duration"])
    common["byte_rate"] = safe_div(common["total_bytes"], common["duration"])
    common["avg_pkt_size"] = safe_div(common["total_bytes"], common["total_pkts"])
    common["dst_port"] = pd.to_numeric(df.get("id.resp_p"), errors="coerce").fillna(0)
    common["src_port"] = pd.to_numeric(df.get("id.orig_p"), errors="coerce").fillna(0)
    return finalize_common(common, df)

#registry of all built-in datasets and their loaders
DATASETS = {"bot_iot": (load_bot_iot, bot_iot_common),
    "ton_iot": (load_ton_iot, ton_iot_common),
    "ciciot2023": (load_ciciot2023, ciciot_common),
    "iot23": (load_iot23, iot23_common)}

'''Generic label unification for unknown datasets'''
def unify_label(raw):
    s = str(raw).strip().lower()
    if s in ("", "nan", "none"):
        return "Benign"
    if any(k in s for k in ("benign", "normal", "legitimate", "background")):
        return "Benign"
    if "ddos" in s:
        return "DDoS" #again order matters (ddos before dos)
    if "mirai" in s:
        return "Mirai"
    if "dos" in s or "denial" in s:
        return "DoS"
    if any(k in s for k in ("gafgyt", "bashlite", "c&c", "cnc", "torii", "okiru", "botnet", "backdoor", "trojan", "ransom", "malware")):
        return "Botnet"
    if any(k in s for k in ("recon", "scan", "probe", "sniff", "discovery", "fingerprint")):
        return "Recon"
    if any(k in s for k in ("brute", "password", "dictionary", "credential")):
        return "BruteForce"
    if "sql" in s or "injection" in s:
        return "Injection"
    if any(k in s for k in ("xss", "web", "http_flood", "upload", "browser")):
        return "Web"
    if any(k in s for k in ("spoof", "mitm", "arp", "man-in")):
        return "Spoofing_MITM"
    if any(k in s for k in ("theft", "exfil", "leak")):
        return "Theft"
    return "Other"
    
'''Normalize column names to lowercase and underscores'''   
def norm_col(df: pd.DataFrame):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

'''Automatic label column detection (with optional override)'''
def detect_label_col(cols: list[str], override: str | None = None):
    if override and override.lower() in cols:
        return override.lower()
    for c in LABEL_COL_CANDIDATES:
        if c in cols:
            return c
    for c in cols:
        if "label" in c or "attack" in c: #last resort, any column with label or attack in its name
            return c
    return None

'''Returns the first column found in candidates'''
def find_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None

'''Reads dataset confuguration from JSON'''
def load_dataset_config():
    if DATASET_CONFIG_PATH.exists():
        try:
            return _json.loads(DATASET_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

'''Saves dataset configuration to JSON'''
def save_dataset_config(config: dict):
    DATASET_CONFIG_PATH.write_text(_json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    
'''Generic CSV reader with chunked sampling to avoid memory issues'''    
def read_csv_sampled(path: Path, per_file: int):
    chunks = []
    got = 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=200000):
        take = min(per_file, len(chunk))
        chunks.append(chunk.sample(n=take, random_state=RANDOM_STATE) if len(chunk) > take else chunk)
        got += take
        if got >= per_file*2:
            break
    return pd.concat(chunks, ignore_index=True) if chunks else pd.read_csv(path)

'''Generic dataset loader for unknown datasets'''
def generic_loader(name: str, folder: Path, label_col=None, glob_pat="*.csv"):
    def loader(cap: int = MAX_ROWS_PER_DATASET):
        files = sorted(p for p in folder.rglob(glob_pat) if "_local_cache" not in str(p) and not p.name.endswith(".pcap.csv"))
        if not files:
            raise FileNotFoundError(f"no {name} CSV found in {folder}")
        per_file = max(cap*2//len(files), 5000)
        df = norm_col(pd.concat([read_csv_sampled(f, per_file) for f in files], ignore_index=True))
        lc = detect_label_col(list(df.columns), label_col)
        if lc is None:
            raise ValueError(f"no label column found in {name} dataset")
        col = df[lc]
        #0/1 label carries no attack (1 becomes Other)
        if pd.api.types.is_numeric_dtype(col) and set(pd.unique(col.dropna())) <= {0, 1}:
            df["label"] = np.where(col.astype(float)>0, "Other", "Benign")
        else:
            df["label"] = col.astype(str).map(unify_label)
        df["binary"] = (df["label"] != "Benign").astype(np.int8)
        #custom datasets also keep their capture time before GENERIC_DROP removes it
        df = attach_timestamp(df, ["timestamp", "ts", "time", "frame.time", "flow_start", "stime"])
        #label column is dropped too
        df = df.drop(columns=[c for c in set(GENERIC_DROP) | {lc} if c in df.columns], errors="ignore")
        df = df.replace([np.inf, -np.inf, "-", "(empty)"], np.nan)
        df = merge_rare_classes(df)
        return stratified_cap(df, cap)
    return loader

        

'''Common feature extraction for unknown datasets'''
def generic_common():
    def common(df: pd.DataFrame):
        proto = find_col(df, PROTO)
        if proto is not None:
            common = proto_onehot(proto)
        else:
            z = np.zeros(len(df), np.int8) #no protocol column
            common = pd.DataFrame({"proto_tcp": z, "proto_udp": z, "proto_icmp": z})
        dur = find_col(df, DUR)
        common["duration"] = pd.to_numeric(dur, errors="coerce") if dur is not None else 0.0
        pk = find_col(df, PKTS)
        if pk is None:
            #no total packet count (reconstruction)
            fp, bp = find_col(df, FWD_PKTS), find_col(df, BWD_PKTS)
            if fp is not None:
                pk = (pd.to_numeric(fp, errors="coerce").fillna(0)+pd.to_numeric(bp, errors="coerce").fillna(0))
        common["total_pkts"] = pd.to_numeric(pk, errors="coerce") if pk is not None else 0.0
        by = find_col(df, BYTES)
        if by is None:
            fb, bb = find_col(df, FWD_BYTES), find_col(df, BWD_BYTES)
            if fb is not None:
                by = (pd.to_numeric(fb, errors="coerce").fillna(0)+pd.to_numeric(bb, errors="coerce").fillna(0))
        common["total_bytes"] = pd.to_numeric(by, errors="coerce") if by is not None else 0.0
        common["pkt_rate"] = safe_div(common["total_pkts"], common["duration"])
        common["byte_rate"] = safe_div(common["total_bytes"], common["duration"])
        common["avg_pkt_size"] = safe_div(common["total_bytes"], common["total_pkts"])
        dpt, spt = find_col(df, DPORT), find_col(df, SPORT)
        common["dst_port"] = pd.to_numeric(dpt, errors="coerce").fillna(0) if dpt is not None else 0
        common["src_port"] = pd.to_numeric(spt, errors="coerce").fillna(0) if spt is not None else 0
        return finalize_common(common, df)
    return common

'''Scans datasets folder for new subfolers (csv datasets)'''
def discover_candidates():
    out: list[dict]= []
    root = dataset_dir()
    if not root.exists():
        return out
    config = load_dataset_config().get("custom", {})
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        nm = sub.name.lower()
        if nm in BUILTIN_DATASETS or nm == "_local_cache":
            continue #builtin datasets are already registered
        csvs = [p for p in sub.rglob("*.csv") if not p.name.endswith(".pcap.csv")]
        if not csvs:
            continue
        try:
            cols = list(norm_col(pd.read_csv(csvs[0], nrows=200)).columns) #200 rows only (column names)
        except Exception:
            cols = []
        c = config.get(nm, {})
        out.append({"name": nm, "folder": str(sub), "n_csv": len(csvs), "columns": cols, "label_col": detect_label_col(cols, c.get("label_col")),
        "enabled": bool(c.get("enabled", False)), "registered": nm in DATASETS}) #user must opt-in explicity
    return out

'''Adds custom datasets in config to the datasets registry'''
def register_custom_datasets():
    config = load_dataset_config().get("custom", {})
    for c in discover_candidates():
        nm = c["name"]
        conf= config.get(nm, {})
        if not conf.get("enabled") or nm in BUILTIN_DATASETS:
            continue
        DATASETS[nm] = (generic_loader(nm, Path(c["folder"]), conf.get("label_col"), conf.get("glob", "*.csv")), generic_common())
    

'''present datasets on disk'''
def available_datasets():
    found = []
    root = dataset_dir()
    for name in DATASETS:
        try:
            probes = BUILTIN_PROBES.get(name)
            if probes: 
                if find(root, probes): #builtin dataset
                    found.append(name)
            else:
                folder = root/name #custom
                if folder.exists() and any(p for p in folder.rglob("*.csv") if not p.name.endswith(".pcap.csv")):
                    found.append(name)
        except Exception:
            pass #broken dataset
    return found


register_custom_datasets()