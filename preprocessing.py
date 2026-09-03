#Libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler #class imbalance handling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (MAX_TRAIN_ROWS, RANDOM_STATE, TEST_SIZE, SPLIT_MODE, TS_COL,
                    SPLIT_AUDIT_PATH, RESULTS_DIR, LOG_SCALE_FEATURES)

import re
import json
import logging
_log = logging.getLogger("preprocessing")

#Prefixes of windowed context columns by loaders.add_window_features
WINDOW_PREFIXES = ("w5_", "w10_")

#Strings "this value is absent"
MISSING_TOKENS = {"", "-", "--", "nan", "none", "null", "na", "n/a", "(empty)", "unknown", "?"}

'''split features and labels'''
def build_xy(df: pd.DataFrame):
    y = df["label"].astype(str) #attack category names
    y_bin = df["binary"].astype(np.int8) #binary labels
    X = df.drop(columns=["label", "binary"]) #variables/features
    for c in X.columns:
        if X[c].dtype != object:
            continue
        coerced = pd.to_numeric(X[c], errors="coerce")
        broke = X[c][coerced.isna() & X[c].notna()] #present, but not a number
        real_categories = broke[~broke.astype(str).str.strip().str.lower().isin(MISSING_TOKENS)]
        #numeric if almost everything parses, or if everything that did not parse was a missing-marker
        if coerced.notna().mean() > 0.95 or real_categories.empty:
            X[c] = coerced
    X = X.replace([np.inf, -np.inf], np.nan) #replace infinities with NaN
    return X, y, y_bin


'''Signed log compression for heavy-tailed features: sign(x) * log1p(|x|)'''
def signed_log1p(X):
    X = np.asarray(X, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)
    return np.sign(X) * np.log1p(np.abs(X))

'''safe cast to float64 with clipping to avoid overflow'''
def to_float64(X):
    X = np.asarray(X, dtype=np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e30, neginf=-1e30)
    return np.clip(X, -1e30, 1e30)


#Columns that hold a TCP/UDP port number, whatever the dataset calls them
PORT_RE = re.compile(r"(^|_|\.)(sport|dport|src_port|dst_port|orig_p|resp_p|port)($|_)", re.I)

#BoT-IoT ships engineered per-IP aggregates whose names end in the same tokens
PER_IP_AGG = ("TnBPSrcIP", "TnBPDstIP", "TnP_PSrcIP", "TnP_PDstIP", "TnP_PerProto", "TnP_Per_Dport",
              "AR_P_Proto_P_SrcIP", "AR_P_Proto_P_DstIP", "AR_P_Proto_P_Sport", "AR_P_Proto_P_Dport",
              "N_IN_Conn_P_SrcIP", "N_IN_Conn_P_DstIP",
              "Pkts_P_State_P_Protocol_P_SrcIP", "Pkts_P_State_P_Protocol_P_DestIP")

'''Columns that really carry a port number: the name matches AND it is not a known aggregate'''
def port_columns(cols):
    return [c for c in cols if PORT_RE.search(str(c)) and str(c) not in PER_IP_AGG]

#Services that actually matter in IoT traffic (MQTT (1883/8883), Modbus (502) and CoAP (5683))
SERVICE_PORTS = (21, 22, 23, 25, 53, 80, 123, 443, 445, 502, 1883, 3389, 5683, 8080, 8883)

'''Turn a port number into things a model can actually use'''
def port_expand(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        #SimpleImputer drops a column that is entirely NaN
        return np.zeros((X.shape[0] if X.ndim else 0, 0), dtype=np.float64)
    cols = []
    for j in range(X.shape[1]):
        p = X[:, j]
        cols.append((p < 0))                       #unknown / not applicable
        cols.append((p >= 0) & (p < 1024))         #well-known
        cols.append((p >= 1024) & (p < 49152))     #registered
        cols.append(p >= 49152)                    #ephemeral, i.e. a client-side port
        for sp in SERVICE_PORTS:
            cols.append(p == sp)
    return np.column_stack(cols).astype(np.float64)

def port_names(_transformer, input_features):
    out = []
    for c in input_features:
        out += [f"{c}_unknown", f"{c}_wellknown", f"{c}_registered", f"{c}_ephemeral"]
        out += [f"{c}_is{sp}" for sp in SERVICE_PORTS]
    return np.asarray(out, dtype=object)

'''build a preprocessor pipeline for numeric and categorical features'''
def make_preprocessor(X: pd.DataFrame):
    #"bool" belongs in the numeric branch
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    #median rather than mean for robustness, most frequent for categorical
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if LOG_SCALE_FEATURES:
        #before the scaler
        num_steps.append(("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
    num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    #handle unknown categories in test set with handle_unknown="ignore"
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    #only include transformers for columns that exist
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    #port columns are fed through the categorical reading above
    port_cols = port_columns(num_cols)
    if port_cols:
        #keep_empty_features
        port_pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("expand", FunctionTransformer(port_expand, feature_names_out=port_names))])
        transformers.append(("port", port_pipe, port_cols))
    ct = ColumnTransformer(transformers, remainder="drop") #unused columns are dropped
    cast = FunctionTransformer(to_float64, feature_names_out="one-to-one")
    return Pipeline([("ct", ct), ("cast", cast)])


'''class-imbalance strategy for training set'''
def apply_imbalance(X_tr, y_tr, strategy: str, random_state = RANDOM_STATE):
    if strategy in ("none", "class_weight"): #models handle class imbalance internally
        return X_tr, y_tr
    if strategy == "undersample": #trims the majority
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "oversample": #duplicates the minority
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == "smote": #k_nearest neighbors never exceeds the samples in the minority class
        counts = pd.Series(y_tr).value_counts()
        k = max(1, min(5, counts.min()-1))
        sampler = SMOTE(k_neighbors=k, random_state=random_state)
    else: #synthetic samples
        raise ValueError(f"Unknown imbalance strategy: {strategy}")
    Xr, yr = sampler.fit_resample(X_tr, y_tr)
    return Xr, yr

'''feature selection with 3 methods variance, correlation, and mutual information'''
def select_features(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr, method: str, k: int = 20):
    if method == "none":
        return X_tr, X_te, list(X_tr.columns)
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    other = [c for c in X_tr.columns if c not in num_cols] #categorical columns are kept
    Xn = X_tr[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0) #clean computations matrix
    if method == "variance":
        keep = [c for c in num_cols if Xn[c].var() > 1e-10]
    elif method == "corr95":
        keep = [c for c in num_cols if Xn[c].var() > 1e-10]
        corr = Xn[keep].corr().abs()
        drop = set()
        cols = list(corr.columns) 
        #every pair with >0.95 keep the first one
        for i, c1 in enumerate(cols):
            if c1 in drop:
                continue
            for c2 in cols[i + 1:]:
                if c2 not in drop and corr.loc[c1, c2] > 0.95:
                    drop.add(c2)
        keep = [c for c in cols if c not in drop]
    elif method == "kbest_mi":
        #k=20 exceeded the column count on IoT-23 and TON-IoT ("select the 20 best")
        k = min(max(5, len(num_cols) // 2), len(num_cols))
        #non-linear mutual information 
        sel = SelectKBest(lambda Xa, ya: mutual_info_classif(Xa, ya, random_state=RANDOM_STATE), k=k).fit(Xn, y_tr)
        keep = [c for c, m in zip(num_cols, sel.get_support()) if m]
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    kept = keep + other
    return X_tr[kept], X_te[kept], kept


_SPLIT_AUDIT: dict = {} #dataset -> what the split actually did, dumped to split_audit.json

'''Record what each split really produced'''
def record_split_audit(dataset, used, ts, y_tr, y_te):
    try:
        tr, te = set(pd.unique(y_tr.astype(str))), set(pd.unique(y_te.astype(str)))
        rec = {"split_mode_requested": SPLIT_MODE, "split_mode_used": used,
               "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
               "classes_train": sorted(tr), "classes_test": sorted(te),
               "missing_from_train": sorted(te - tr), #unlearnable classes
               "missing_from_test": sorted(tr - te)}
        if used == "temporal" and ts is not None and ts.notna().any():
            rec["train_time_range"] = [float(ts.loc[y_tr.index].min()), float(ts.loc[y_tr.index].max())]
            rec["test_time_range"] = [float(ts.loc[y_te.index].min()), float(ts.loc[y_te.index].max())]
            #no overlap means the protocol really is chronological
            rec["time_ranges_disjoint"] = bool(rec["train_time_range"][1] <= rec["test_time_range"][0])
        if rec["missing_from_train"]:
            _log.warning("%s [%s]: %d class(es) absent from train and therefore unlearnable: %s",
                         dataset or "dataset", used, len(rec["missing_from_train"]), rec["missing_from_train"])
        rec["recorded_by"] = Path(sys.argv[0]).stem or "python"
        _SPLIT_AUDIT[str(dataset or "unnamed")] = rec
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        #every phase runs in its own process and used to overwrite the whole file
        merged = {}
        if SPLIT_AUDIT_PATH.exists():
            try:
                merged = json.loads(SPLIT_AUDIT_PATH.read_text(encoding="utf-8"))
            except Exception:
                merged = {}
        merged.update(_SPLIT_AUDIT)
        tmp = SPLIT_AUDIT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(merged, indent=1, default=str), encoding="utf-8")
        try:
            tmp.replace(SPLIT_AUDIT_PATH)
        except OSError:
            pass #a locked audit file must never stop an experiment
    except Exception as e:
        _log.warning("split audit failed (%s); the split itself is unaffected", e)

'''What the split actually produced for a dataset, so every downstream table can carry it'''
def split_meta(dataset, y_train=None, y_test=None):
    r = _SPLIT_AUDIT.get(str(dataset), {})
    out = {"split_mode": r.get("split_mode_used") if r else None, "n_test_classes": None, "classes_missing_from_train": None, "evaluable": None}
    if y_train is None or y_test is None:
        return out
    tr = set(pd.unique(pd.Series(y_train).astype(str)))
    te = set(pd.unique(pd.Series(y_test).astype(str)))
    missing = sorted(te - tr)
    out["n_test_classes"] = len(te)
    out["classes_missing_from_train"] = missing
    #present in train is not the same as learnable
    counts = pd.Series(y_train).astype(str).value_counts()
    te_counts = pd.Series(y_test).astype(str).value_counts()
    #two ways a present class is still not learnable
    def starving(c):
        n_tr, n_te = int(counts.get(c, 0)), int(te_counts.get(c, 0))
        if n_tr == 0:
            return False #already reported as missing_from_train
        return n_tr < 100 or (n_tr < 5000 and n_tr < 0.05 * n_te)
    starved = sorted(c for c in te if starving(c))
    out["min_train_per_class"] = int(counts.reindex(sorted(te)).fillna(0).min()) if te else None
    out["classes_undertrained"] = starved
    out["evaluable"] = bool(len(te) >= 2 and not missing)
    out["fully_learnable"] = bool(out["evaluable"] and not starved)
    return out

'''train-test split with stratification and optional trimming of training set'''
def split(X, y, y_bin, dataset=None):
    #TS_COL is metadata, never a feature
    ts = None
    if TS_COL in X.columns:
        ts = pd.to_numeric(X[TS_COL], errors="coerce")
        X = X.drop(columns=[TS_COL])
    #Deduplicate on the FLOW columns only
    _dedup_on = [c for c in X.columns if not str(c).startswith(WINDOW_PREFIXES)] or list(X.columns)
    _keep = ~X[_dedup_on].assign(_label=y.values).duplicated()
    if len(_dedup_on) != X.shape[1]:
        _log.info("dedup on %d flow columns (%d windowed columns excluded): %d -> %d rows", len(_dedup_on), X.shape[1] - len(_dedup_on), len(X), int(_keep.sum()))
    X, y, y_bin = X[_keep], y[_keep], y_bin[_keep]
    if ts is not None:
        ts = ts[_keep]
    #temporal needs a usable clock
    mode = SPLIT_MODE if SPLIT_MODE in ("random", "temporal") else "random"
    used = mode
    if mode == "temporal" and (ts is None or ts.notna().sum() < 10 or ts.nunique(dropna=True) < 2):
        used = "random"
        _log.warning("%s: temporal split requested but no usable timestamp -> falling back to random", dataset or "dataset")
    if used == "temporal":
        #oldest TEST_SIZE-complement in train, newest in test
        order = ts.fillna(ts.min() - 1).sort_values(kind="mergesort").index #stable: ties keep file order
        cut = int(round(len(order) * (1 - TEST_SIZE)))
        tr_idx, te_idx = order[:cut], order[cut:]
        #safety net: a chronological cut can leave train with a single class (or none)
        if y.loc[tr_idx].nunique() < 2 or y.loc[te_idx].nunique() < 1:
            _log.warning("%s: temporal cut leaves %d class(es) in train -> falling back to random", dataset or "dataset", y.loc[tr_idx].nunique())
            used = "random"
        else:
            X_tr, X_te = X.loc[tr_idx], X.loc[te_idx]
            y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
            y_bin_tr, y_bin_te = y_bin.loc[tr_idx], y_bin.loc[te_idx]
            record_split_audit(dataset, used, ts, y_tr, y_te)
    if used == "random":
        X_tr, X_te, y_tr, y_te, y_bin_tr, y_bin_te = train_test_split(X, y, y_bin, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y) #same class propotions in train and test
        record_split_audit(dataset, used, ts, y_tr, y_te)
    #The floor min(len(s), 1000) keeps small classes whole large classes downsampled proportionally
    if len(X_tr) > MAX_TRAIN_ROWS:
        before = y_tr.value_counts(normalize=True)
        n_before = len(X_tr)
        idx = (pd.Series(range(len(X_tr)), index=X_tr.index).groupby(y_tr, group_keys=False).apply(lambda s: s.sample( n=max(int(len(s)*MAX_TRAIN_ROWS/len(X_tr)), min(len(s), 1000)), random_state=RANDOM_STATE)))
        X_tr = X_tr.loc[idx.index]
        y_tr = y_tr.loc[idx.index]
        y_bin_tr = y_bin_tr.loc[idx.index]
        after = y_tr.value_counts(normalize=True)
        shift = float((after - before).reindex(before.index).abs().max())
        _log.info("train cap %d -> %d rows; max class-share shift from the min-1000 floor: %+.4f", n_before, len(X_tr), shift)
        if shift > 0.01:
            _log.warning("the capped training set is mildly rebalanced (max class-share shift %.4f): " "the 'none' imbalance strategy is not a pure unbalanced baseline", shift)
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in X_tr.select_dtypes(include="object").columns:
        top = X_tr[c].value_counts().nlargest(30).index
        #drop single value columns   
        X_tr[c] = X_tr[c].where(X_tr[c].isin(top), "RARE").fillna("MISSING").astype(str)
        X_te[c] = X_te[c].where(X_te[c].isin(top), "RARE").fillna("MISSING").astype(str)
    nunique = X_tr.nunique(dropna=False)
    keep = list(nunique[nunique > 1].index)
    X_tr, X_te = X_tr[keep], X_te[keep]
    return X_tr, X_te, y_tr, y_te, y_bin_tr, y_bin_te