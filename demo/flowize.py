#Libraries
import re
import numpy as np
import pandas as pd

FEATURES = ["duration", "total_pkts", "total_bytes", "pkt_rate", "byte_rate", "avg_pkt_size", "proto_tcp", "proto_udp", "proto_icmp", "dst_port", "src_port"]
PORTS = re.compile(r"(\d+)\s*->\s*(\d+)")

"""Packet list carries the ports inside Info"""
def _ports_from_info(info: str):
    m = PORTS.search(str(info))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

"""The columns a saved pipeline was fitted on, or None if it does not record them."""
def model_columns(model):
    cols = getattr(model, "feature_names_in_", None)
    return None if cols is None else [str(c) for c in cols]

"""A native_<dataset>.csv table (only empty cells are missing)"""
def read_native(path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""]).set_index("flow_id")

    
"""A per-dataset model's own fields, typed the way its preprocessor was fitted"""
def native_input(model, table: pd.DataFrame) -> pd.DataFrame:
    cols = model_columns(model)
    ct = model.named_steps["pre"].named_steps["ct"]
    cat = {c for name, _, cs in ct.transformers_ if name == "cat" for c in cs}
    X = table[cols].copy()
    for c in cols:
        X[c] = X[c].astype(str) if c in cat else pd.to_numeric(X[c], errors="coerce")
    return X

def packets_to_flows(pkts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid, g in pkts.groupby("flow_id", sort=True):
        g = g.sort_values("Time")
        dur   = float(g["Time"].max() - g["Time"].min())
        npk   = int(len(g))
        nby   = float(g["Payload"].sum())
        proto = str(g["Protocol"].iloc[0]).upper()
        #a real exporter (Zeek, nfstream) emits ports as fields
        first = g.iloc[0]
        if "SrcPort" in g.columns and pd.notna(first.get("SrcPort")):
            sp, dp = int(first["SrcPort"]), int(first["DstPort"])
        else:
            sp, dp = _ports_from_info(first["Info"])
        rows.append({"flow_id": fid, "duration": dur,"total_pkts": npk,"total_bytes": nby,
            #a flow with no elapsed time has no rate
            "pkt_rate":  npk / dur if dur > 0 else 0.0,"byte_rate": nby / dur if dur > 0 else 0.0,
            "avg_pkt_size": nby / npk if npk else 0.0,"proto_tcp":  int(proto == "TCP"),
            "proto_udp":  int(proto == "UDP"),"proto_icmp": int(proto == "ICMP"), "dst_port": dp,"src_port": sp})
    return pd.DataFrame(rows).set_index("flow_id").sort_index()
