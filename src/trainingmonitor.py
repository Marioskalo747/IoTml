#Libraries
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px #high level plotting
import plotly.graph_objects as go #manual composition
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import ARCHIVES_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR

TEMPORAL_RESULTS_DIR = RESULTS_DIR.with_name(RESULTS_DIR.name + "_temporal")
TEMPORAL_MODELS_DIR = MODELS_DIR.with_name(MODELS_DIR.name + "_temporal")
try:
    import loaders as L #optional
except Exception as _e:
    L, LOADERS_ERR = None, _e

st.set_page_config(page_title="IoT IDS · Pipeline", layout="wide",
                   initial_sidebar_state="expanded")
#palette and theme definitions
LIGHT = dict(bg="#e8ecf2", surface="#ffffff", surface2="#f4f6f9", line="#d5dbe4",
    ink="#0a1526", ink2="#39445a", ink3="#6a7488", #3 levels of text
    run="#0f5bff", run_bg="#e5edff", run_line="#b3ccff", #running
    ok="#0a7d55", ok_bg="#e4f5ee", ok_line="#a8dfc7", #completed
    bad="#c02626", bad_bg="#fdeaea", bad_line="#f5bfbf", #failed
    idle="#5c6779", idle_bg="#eef1f6", idle_line="#d5dbe4", #pending
    grid="rgba(10,21,38,.09)", axis="rgba(10,21,38,.22)",
    plot="plotly_white", scale="Blues",
    seq=["#0f5bff", "#00a6a6", "#7a5af8", "#e8873a", "#0a7d55", #categorical
         "#c02626", "#2d9cdb", "#b14aed", "#8a6d3b", "#5c6779"],
)
#same keys as light
DARK = dict(bg="#11151c", surface="#1a202b", surface2="#151a23", line="#2b3444",
    ink="#dce3ed", ink2="#a3adbe", ink3="#78839a",
    run="#5c8dff", run_bg="#16233d", run_line="#2a4270",
    ok="#4cc38a", ok_bg="#12291f", ok_line="#1f4a35",
    bad="#ef6f6f", bad_bg="#2a1719", bad_line="#5b2b2e",
    idle="#8792a6", idle_bg="#1a202b", idle_line="#2b3444",
    grid="rgba(220,227,237,.10)", axis="rgba(220,227,237,.20)",
    plot="plotly_dark", scale="Blues",
    seq=["#5c8dff", "#4ecdc4", "#a78bfa", "#f0a868", "#4cc38a",
         "#ef6f6f", "#63b3ed", "#c77dff", "#d6bd7a", "#8792a6"],
)

#containers reserved for sidebar sections
side_nav = st.sidebar.container()
side_view = st.sidebar.container()
side_refresh = st.sidebar.container()
side_look = st.sidebar.container()

with side_look:
    st.markdown("<div class='side-t'>Appearance</div>", unsafe_allow_html=True)
    mode = st.radio("Mode", ["Light", "Dark"], horizontal=True,
                    label_visibility="collapsed")
T = LIGHT if mode == "Light" else DARK #decision on theme colors
PALETTE = T["seq"]

#1st CSS block (theme in css)
st.markdown(f"""
<style>
:root{{
  --bg:{T['bg']}; --surface:{T['surface']}; --surface-2:{T['surface2']}; --line:{T['line']};
  --ink:{T['ink']}; --ink-2:{T['ink2']}; --ink-3:{T['ink3']};
  --run:{T['run']}; --run-bg:{T['run_bg']}; --run-line:{T['run_line']};
  --ok:{T['ok']};   --ok-bg:{T['ok_bg']};   --ok-line:{T['ok_line']};
  --bad:{T['bad']}; --bad-bg:{T['bad_bg']}; --bad-line:{T['bad_line']};
  --idle:{T['idle']};--idle-bg:{T['idle_bg']};--idle-line:{T['idle_line']};}}
</style>
""", unsafe_allow_html=True)

#2nd CSS block (layout and components)(typography, headings, phases, progress bars, activity car etc)
st.markdown("""
<style>
.stApp{background:var(--bg);}
[data-testid="stSidebar"]{background:var(--surface-2);border-right:1px solid var(--line);}
.block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:1500px;}
#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden;}

h1,h2,h3,h4,h5{color:var(--ink) !important;font-weight:600;letter-spacing:-.011em;}
[data-testid="stMarkdownContainer"]{color:var(--ink);}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{color:var(--ink-2);}
[data-testid="stWidgetLabel"] p, label p, .stRadio label, .stSelectbox label{color:var(--ink-2) !important;font-size:.82rem;}
[data-testid="stExpander"] summary p{color:var(--ink) !important;}
.stAlert p{color:var(--ink-2) !important;}
code, .stCode, pre{color:var(--ink) !important;}
[data-testid="stSidebar"] .stCode > div{background:var(--bg) !important;}

.sec{margin:1.7rem 0 .35rem;}
.sec:first-of-type{margin-top:.2rem;}
.sec-t{font-size:1rem;font-weight:600;color:var(--ink);}
.sec-s{font-size:.79rem;color:var(--ink-3);margin-top:.14rem;line-height:1.5;}
.rule{height:1px;background:var(--line);margin:.5rem 0 1rem;}
.hint{font-size:.78rem;color:var(--ink-3);line-height:1.55;margin:.1rem 0 .7rem;}

.bdg{display:inline-block;padding:.09rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600;letter-spacing:.02em;white-space:nowrap;}
.bdg-ok{background:var(--ok-bg);color:var(--ok) !important;border:1px solid var(--ok-line);}
.bdg-run{background:var(--run-bg);color:var(--run) !important;border:1px solid var(--run-line);}
.bdg-idle{background:var(--idle-bg);color:var(--idle) !important;border:1px solid var(--idle-line);}
.bdg-bad{background:var(--bad-bg);color:var(--bad) !important;border:1px solid var(--bad-line);}

.ph{display:flex;align-items:center;gap:.8rem; padding:.5rem .85rem;border-left:3px solid var(--line);background:var(--surface); margin-bottom:2px;border-radius:0 6px 6px 0;}
.ph-done{border-left-color:var(--ok);}
.ph-run{border-left-color:var(--run);background:var(--run-bg);}
.ph-bad{border-left-color:var(--bad);background:var(--bad-bg);}
.ph-n{width:1.5rem;color:var(--ink-3);font-size:.76rem;font-variant-numeric:tabular-nums;}
.ph-name{flex:1;font-size:.88rem;color:var(--ink);}
.ph-time{font-size:.76rem;color:var(--ink-3);font-variant-numeric:tabular-nums; min-width:5rem;text-align:right;}

.crumbs{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem;margin:.15rem 0 1rem;}
.cr{font-size:.77rem;padding:.2rem .62rem;border-radius:6px; background:var(--idle-bg);color:var(--ink-3);border:1px solid transparent;}
.cr-past{color:var(--ok) !important;background:var(--ok-bg);}
.cr-now{color:var(--run) !important;background:var(--run-bg); border-color:var(--run-line);font-weight:600;}
.cr-sep{color:var(--ink-3);font-size:.66rem;opacity:.55;}

.kpi{border:1px solid var(--line);border-radius:9px;padding:.65rem .85rem; background:var(--surface);height:100%;}
.kpi-l{font-size:.68rem;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;}
.kpi-v{font-size:1.3rem;font-weight:600;color:var(--ink);margin-top:.15rem; font-variant-numeric:tabular-nums;line-height:1.2;}
.kpi-s{font-size:.71rem;color:var(--ink-3);margin-top:.08rem;}

.bar{height:6px;border-radius:3px;background:var(--idle-bg);overflow:hidden; margin:.15rem 0 .3rem;border:1px solid var(--line);}
.bar-f{height:100%;background:var(--run);transition:width .3s ease;}
.bar-f.ok{background:var(--ok);}
.bar-cap{display:flex;justify-content:space-between;font-size:.74rem; color:var(--ink-3);font-variant-numeric:tabular-nums;}

.act{border:1px solid var(--line);border-radius:9px;padding:.85rem 1rem;background:var(--surface);}
.act-l{font-size:.68rem;color:var(--ink-3);text-transform:uppercase;letter-spacing:.06em;}
.act-v{font-size:1.02rem;color:var(--ink);margin-top:.2rem;font-weight:500;}
.act-sub{font-size:.8rem;color:var(--ink-2);margin-top:.3rem;line-height:1.5;}
.mono{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:.8rem;}

.note{border-left:3px solid var(--line);padding:.55rem .85rem;border-radius:0 6px 6px 0; font-size:.83rem;line-height:1.5;background:var(--surface);color:var(--ink-2);}
.note-run{border-left-color:var(--run);background:var(--run-bg);color:var(--run) !important;}
.note-ok{border-left-color:var(--ok);background:var(--ok-bg);color:var(--ok) !important;}
.note-bad{border-left-color:var(--bad);background:var(--bad-bg);color:var(--bad) !important;}
.note-warn{border-left-color:var(--idle);background:var(--idle-bg);color:var(--ink-2) !important;}

.ev{font-family:ui-monospace,Consolas,monospace;font-size:.76rem;color:var(--ink-2); padding:.24rem .6rem;border-bottom:1px solid var(--line);background:var(--surface);}
.ev:first-child{border-radius:6px 6px 0 0;}
.ev:last-child{border-bottom:none;border-radius:0 0 6px 6px;}
.ev-t{color:var(--ink-3);margin-right:.5rem;}

.side-t{font-size:.68rem;color:var(--ink-3);text-transform:uppercase; letter-spacing:.07em;margin:1rem 0 .25rem;font-weight:600;}
[data-testid="stSidebar"] .stRadio [role="radiogroup"]{gap:.05rem;}
[data-testid="stSidebar"] .stRadio label{padding:.12rem 0;}
</style>
""", unsafe_allow_html=True)

#'''Consistent style for figures and charts'''
def style_fig(fig, height=None, legend=True):
    fig.update_layout(template=T["plot"],font=dict(family="system-ui,-apple-system,Segoe UI,sans-serif", size=12, color=T["ink2"]),
        margin=dict(l=6, r=6, t=32 if legend else 12, b=6),plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", #transparent
        colorway=PALETTE, hoverlabel=dict(font_size=12),legend=(dict(orientation="h", yanchor="bottom", y=1.02, x=0,title_text="", 
        font=dict(size=11)) if legend else {})) #horizontal legend above the figure
    #horizontal grid lines
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor=T["axis"],ticks="outside", tickfont=dict(size=11, color=T["ink3"]), title_font=dict(size=11, color=T["ink3"]))
    fig.update_yaxes(gridcolor=T["grid"], showline=False, zeroline=False,tickfont=dict(size=11, color=T["ink3"]),title_font=dict(size=11, color=T["ink3"]))
    if height:
        fig.update_layout(height=height)
    return fig

#'''Shorthand style and rendering'''
def chart(fig, height=None, legend=True):
    st.plotly_chart(style_fig(fig, height, legend), width="stretch", config={"displaylogo": False,"modeBarButtonsToRemove": ["lasso2d", "select2d","autoScale2d"]})

#Translation dictionaries for phases, stages, badges, and labels
PHASE_LABEL = {"run_all": "Baseline training", "tuning": "Hyperparameter search",
    "extended_study": "Ablation study", "deep_study": "Deep models",
    "anomaly_study": "Anomaly detection", "adaptation_study": "Domain adaptation",
    "deployment_bench": "Deployment benchmark", "extended_figures": "Figures",
    "report_assets": "Report tables",
    "validation_stats": "Statistical validation", "domain_shift_study": "Domain shift",
    "anomaly_sweep": "Anomaly ROC sweep", "leakage_ablation": "Feature-group ablation",
    "operating_points": "Operating points", "cost_sensitive": "Cost-sensitive training",
    "specialist_study": "Cascade and specialists", "sanity_check": "Sanity check",
    #second protocol, own results directory
    "temporal_run_all": "Temporal: training", "temporal_report_assets": "Temporal: tables",
    "temporal_validation_stats": "Temporal: statistics", "temporal_seed_study": "Temporal: seeds",
    "temporal_sanity_check": "Temporal: sanity check",
    "seed_study": "Seed stability"}

#phase stages of progress
PHASE_STAGES = {"run_all": ["loading", "training", "LODO", "finished"],
    "tuning": ["loading", "tuning", "refit", "finished"],
    "extended_study": ["loading", "imbalance", "feature_selection","learning_curve", "cv", "finished"],
    "deep_study": ["loading", "ablation", "learning_curve", "tuning", "finished"],
    "anomaly_study": ["loading", "in_domain", "cross_domain", "finished"],
    "adaptation_study": ["loading", "adaptation", "finished"],
    "deployment_bench": ["loading", "benchmark", "finished"],
    "extended_figures": ["loading", "backfill", "heatmaps", "curves", "shap", "finished"],
    "report_assets": ["loading", "tables", "figures", "finished"],
    "validation_stats": ["bootstrap_ci", "selection_bias", "finished"],
    "domain_shift_study": ["loading", "source_id", "lodo_raw", "lodo_norm", "finished"],
    "anomaly_sweep": ["in_domain", "cross_domain", "finished"],
    "leakage_ablation": ["ablation", "finished"],
    "seed_study": ["training", "finished"]}

STAGE_LABEL = {"starting": "Starting", "loading": "Load", "training": "Train", "LODO": "Cross-dataset", "tuning": "Optuna search", "refit": "Refit best",
    "imbalance": "Imbalance", "feature_selection": "Feature selection", "learning_curve": "Learning curves", "cv": "Cross-validation",
    "ablation": "Ablation", "in_domain": "In-domain", "cross_domain": "Zero-day", "adaptation": "Adaptation", "benchmark": "Latency", "backfill": "Backfill",
    "heatmaps": "Heatmaps", "curves": "ROC and PR", "shap": "Attribution","tables": "Tables", "figures": "Figures", "finished": "Done","bootstrap_ci": "Bootstrap CI", 
    "selection_bias": "Selection bias","source_id": "Source ID", "lodo_raw": "LODO raw", "lodo_norm": "LODO normalised","ablation": "Ablation"}
#"findings" is what runfull records for an advisory phase that exited non-zero
STATUS_BADGE = {"done": ("bdg-ok", "done"), "running": ("bdg-run", "running"),"failed": ("bdg-bad", "failed"),
                "findings": ("bdg-run", "findings"), "pending": ("bdg-idle", "pending")}
PHASE_CLASS = {"done": "ph-done", "running": "ph-run", "failed": "ph-bad", "findings": "ph-done", "pending": ""}
COMPLETED_STATUSES = ("done", "findings") #both mean the phase is over

IMB_LABEL = {"none": "No handling", "class_weight": "Class weights","undersample": "Undersampling", "oversample": "Oversampling","smote": "SMOTE"}
FS_LABEL = {"none": "All features", "variance": "Variance filter","corr95": "Correlation > .95", "kbest_mi": "Mutual information (k=20)"}
FAMILY_LABEL = {"linear": "Linear", "bayes": "Naive Bayes", "tree": "Decision tree","ensemble": "Bagging ensembles", "boosting": "Gradient boosting",
                "instance": "Nearest neighbours", "svm": "Support vector","neural": "Neural networks"}


#'''Cached JSON reader'''
@st.cache_data(show_spinner=False)
def readjs(path_str, mtime):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))

#'''Load a JSON file from results or archives'''
def load(name, cached=True):
    p = res_dir / name
    if not p.exists():
        return None # File does not exist
    try:
        return readjs(str(p), p.stat().st_mtime) if cached else json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None #half-written or corrupted file

#'''Duration and timestamp formatting'''
def fmt_dur(s):
    if s is None:
        return "—"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"

#'''ISO timestamp formatting'''
def fmt_ts(iso):
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m %H:%M")
    except Exception:
        return str(iso)

#'''Section header'''
def sec(title, sub=""):
    st.markdown(f"<div class='sec'><div class='sec-t'>{title}</div>" f"{f'<div class=sec-s>{sub}</div>' if sub else ''}</div>" f"<div class='rule'></div>", unsafe_allow_html=True)

#'''Hint text'''
def hint(text):
    st.markdown(f"<div class='hint'>{text}</div>", unsafe_allow_html=True)

#'''KPI display'''
def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='kpi-l'>{label}</div>" f"<div class='kpi-v'>{value}</div>" f"{f'<div class=kpi-s>{sub}</div>' if sub else ''}</div>", unsafe_allow_html=True)

#'''Progress bar'''
def bar(pct, left="", right="", done=False):
    pct = max(0.0, min(float(pct), 100.0)) #clamp to [0, 100]
    st.markdown(f"<div class='bar'><div class='bar-f{' ok' if done else ''}' " f"style='width:{pct}%'></div></div>" f"<div class='bar-cap'><span>{left}</span><span>{right}</span></div>", unsafe_allow_html=True)

#'''Note text (running, ok, bad, warn)'''
def note(text, kind="warn"):
    st.markdown(f"<div class='note note-{kind}'>{text}</div>", unsafe_allow_html=True)

#'''Stage crumbs '''
def crumbs(stages, current):
    idx = stages.index(current) if current in stages else -1
    parts = []
    for i, k in enumerate(stages):
        cls = "cr-past" if (idx >= 0 and i < idx) else ("cr-now" if i == idx else "")
        parts.append(f"<span class='cr {cls}'>{STAGE_LABEL.get(k, k)}</span>")
    st.markdown("<div class='crumbs'>" + "<span class='cr-sep'>›</span>".join(parts) + "</div>", unsafe_allow_html=True)

#'''Empty section note'''
def empty(msg="Nothing recorded for this section yet."):
    note(msg, "warn")

# Column configurations for the results table
NUMCOLS = {"dataset": st.column_config.TextColumn("Dataset"),
    "task": st.column_config.TextColumn("Task"),
    "model": st.column_config.TextColumn("Model"),
    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
    "f1_macro": st.column_config.ProgressColumn("F1 macro", format="%.4f", min_value=0, max_value=1),
    "f1_weighted": st.column_config.NumberColumn("F1 weighted", format="%.4f"),
    "mcc": st.column_config.NumberColumn("MCC", format="%.4f"),
    "train_time_s": st.column_config.NumberColumn("Train (s)", format="%.1f"),
    "train_rows": st.column_config.NumberColumn("Train rows", format="localized"),
    "test_rows": st.column_config.NumberColumn("Test rows", format="localized")}

PAGES = ["Run", "Overview", "Models", "Tuning", "Ablations", "Generalisation", "Files"]
GROUP = {"Run": "Monitoring", "Overview": "Results", "Files": "Archive"}

with side_nav:
    st.markdown("<div class='side-t'>Monitoring</div>", unsafe_allow_html=True)
    #current run or archived runs
    page = st.radio("Page", PAGES, label_visibility="collapsed")

with side_view:
    st.markdown("<div class='side-t'>Run</div>", unsafe_allow_html=True)
    archives = sorted((p.name for p in ARCHIVES_DIR.glob("run_*") if p.is_dir()), reverse=True) if ARCHIVES_DIR.exists() else []
    view = st.selectbox("Run", ["current"] + archives, label_visibility="collapsed", format_func=lambda v: "Current run" if v == "current" else v.replace("run_", "Archive "))

with side_refresh:
    st.markdown("<div class='side-t'>Refresh</div>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-refresh while running", value=True)
    refresh_sec = st.slider("Interval", 3, 30, 5, label_visibility="collapsed", format="%ds", disabled=not auto_refresh)

#current run or archived run directories
res_dir = RESULTS_DIR if view == "current" else ARCHIVES_DIR / view / "results"
mod_dir = MODELS_DIR if view == "current" else ARCHIVES_DIR / view / "models"
fig_dir = res_dir / "figures"
meta = load("run_meta.json", cached=False) if view == "current" else None
#a temporal_* phase writes its progress into results_temporal/, so read it from there while one is
#running. run_meta.json always stays in the main directory, written by the parent process
_running = next((n for n, v in (meta or {}).get("phases", {}).items()
                 if v.get("status") == "running"), "") if meta else ""
if view == "current" and _running.startswith("temporal_") and (TEMPORAL_RESULTS_DIR / "progress.json").exists():
    try:
        prog = json.loads((TEMPORAL_RESULTS_DIR / "progress.json").read_text(encoding="utf-8"))
    except Exception:
        prog = None
else:
    prog = load("progress.json", cached=False) #changes frequently
results = load("all_results.json") or [] #cached
exp = [r for r in results if not str(r.get("dataset", "")).startswith("LODO")]
#LODO has its own selection
df_exp = pd.DataFrame([{k: r.get(k) for k in("dataset", "task", "model", "family", "accuracy", "f1_macro", "f1_weighted", "mcc", "train_time_s", "train_rows", "test_rows")} for r in exp])
h1, h2 = st.columns([3, 1])
h1.markdown("IoT IDS — Pipeline")
h1.markdown(f"<div class='sec-s'>{page} · "f"{'current run' if view == 'current' else view} · "f"{datetime.now():%H:%M:%S}</div>", unsafe_allow_html=True)
if meta and meta.get("run_id"):
    h2.markdown(f"<div style='text-align:right;padding-top:1.5rem'>"f"<span class='mono' style='color:var(--ink-3)'>{meta['run_id']}</span>"f"</div>", unsafe_allow_html=True)

#'''Main page rendering logic'''
def page_run():
    running_phase = None
    if meta and meta.get("phases"):
        phases = meta["phases"]
        order = list(phases)
        done = sum(1 for v in phases.values() if v.get("status") in COMPLETED_STATUSES)
        running_phase = next((n for n in order if phases[n].get("status") == "running"), None)
        failed = [n for n in order if phases[n].get("status") == "failed"]
        total = len(order)
        sec("Pipeline", f"{total} phases, executed one after another by runfull.py")
        #status summary (done, running, failed, pending)
        findings = [n for n in order if phases[n].get("status") == "findings"]
        if meta.get("status") == "done" and not failed:
            note(f"All {total} phases completed.", "ok")
        elif failed:
            note(f"Stopped after {done} of {total} phases. Failed: " f"{', '.join(PHASE_LABEL.get(f, f) for f in failed)}", "bad")
        elif running_phase:
            note(f"Phase {order.index(running_phase) + 1} of {total} — " f"{PHASE_LABEL.get(running_phase, running_phase)}", "run")
        else:
            note(f"Pending — {done} of {total} phases completed.", "warn")

        bar(100 * done / total, f"{done} of {total} phases", f"{100 * done / total:.0f}%", done=(done == total))
        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        #phase details (number, name, status badge, start time, elapsed time)
        rows = []
        for i, (name, info) in enumerate(phases.items(), 1):
            stt = info.get("status", "pending")
            bcls, btxt = STATUS_BADGE.get(stt, STATUS_BADGE["pending"])
            rows.append(f"<div class='ph {PHASE_CLASS.get(stt, '')}'>"
                        f"<span class='ph-n'>{i:02d}</span>"
                        f"<span class='ph-name'>{PHASE_LABEL.get(name, name)}</span>"
                        f"<span class='bdg {bcls}'>{btxt}</span>"
                        f"<span class='ph-time'>{fmt_ts(info.get('started_at'))}</span>"
                        f"<span class='ph-time'>{fmt_dur(info.get('elapsed_s'))}</span></div>")
        st.markdown("".join(rows), unsafe_allow_html=True) #markdown call all rows
        #Stall detection
        if running_phase:
            lp = ROOT / phases[running_phase].get("log", "")
            if lp.exists():
                age = time.time() - lp.stat().st_mtime
                if age > 2700: #45 minutes
                    note(f"The phase log has not changed for {fmt_dur(age)}. " f"Check that the process is alive.", "bad")
                elif age > 900: #15 minutes (slow search)
                    note(f"Quiet for {fmt_dur(age)} — expected during long searches " f"that only write once per combination.", "warn")

    if not prog:
        sec("Progress")
        empty("No progress file yet.")
        return
    stage = prog.get("stage", "")
    age = time.time() - prog.get("updated", 0) #seconds since last write
    finished = stage == "finished"
    phase = prog.get("phase") or running_phase or ""
    stages = prog.get("stages") or PHASE_STAGES.get(phase, PHASE_STAGES["run_all"])
    sec(PHASE_LABEL.get(phase, "Last reported activity"),"Live values written by the phase that is running" if phase else "Left over from the last phase that reported progress")

    if stage in stages:
        crumbs(stages, stage)

    dn, tot = prog.get("done", 0), prog.get("total", 0) or 1
    c1, c2 = st.columns([2, 1])
    #Now: dataset , task , model
    where = " · ".join(x for x in (prog.get("dataset"), prog.get("task"), prog.get("model")) if x)
    c1.markdown(f"<div class='act'><div class='act-l'>Now</div>"
                f"<div class='act-v'>{where or '—'}</div>"
                f"<div class='act-sub'>{prog.get('substage') or prog.get('message') or ''}"
                f"</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='act'><div class='act-l'>Stage</div>"
                f"<div class='act-v'>{STAGE_LABEL.get(stage, stage or '—')}</div>"
                f"<div class='act-sub'>updated {fmt_dur(age)} ago</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
    bar(prog.get("pct", 100 * dn / tot), f"{dn:,} of {tot:,} steps", f"{prog.get('pct', 0):.1f}%", done=finished)

    k = st.columns(4)
    kpi(k[0], "Elapsed", fmt_dur(prog.get("elapsed_s")))
    kpi(k[1], "Remaining", "—" if finished else fmt_dur(prog.get("eta_s")))
    kpi(k[2], "Steps left", f"{max(tot - dn, 0):,}")
    kpi(k[3], "Heartbeat", "idle" if finished else fmt_dur(age), "phase finished" if finished else "since last write")

    if not finished and age > 1800: #30 minutes no progress written
        note(f"No progress written for {fmt_dur(age)}.", "bad")
    #details of the current phase into progress.json
    det = prog.get("details") or {}
    t = det.get("tuning")
    if t:
        sec("Search in progress")
        q = st.columns(4)
        kpi(q[0], "Trial", f"{t.get('trial', 0)} / {t.get('trials', '—')}")
        best = t.get("best_cv_f1_macro")
        kpi(q[1], "Best F1 macro", f"{best:.4f}" if isinstance(best, (int, float)) else "—")
        kpi(q[2], "Rows searched", f"{t.get('tune_rows', 0):,}")
        kpi(q[3], "Folds", str(t.get("cv_folds", "—"))) #Nan/None 

    prep = det.get("preprocessing")
    if prep:
        #class distribution
        sec("Dataset being processed")
        p = st.columns(4)
        kpi(p[0], "Dataset", prep.get("dataset", "—"))
        tr = prep.get("train_rows", prep.get("rows_loaded"))
        kpi(p[1], "Train rows", f"{tr:,}" if isinstance(tr, int) else "—")
        te = prep.get("test_rows")
        kpi(p[2], "Test rows", f"{te:,}" if isinstance(te, int) else "—")
        kpi(p[3], "Features", str(prep.get("n_features", "—")), f"{prep.get('n_classes', '—')} classes")
        if prep.get("classes"):
            cc = (pd.DataFrame(prep["classes"].items(), columns=["Class", "Rows"]).sort_values("Rows"))
            f = px.bar(cc, x="Rows", y="Class", orientation="h", text="Rows")
            f.update_traces(marker_color=PALETTE[0], texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
            f.update_xaxes(type="log", title="rows (log scale)")
            f.update_yaxes(title="")
            chart(f, height=60 + 32 * len(cc), legend=False) #scaling height with number of classes

    if prog.get("recent"):
        sec("Recent events")
        html = []
        for line in reversed(prog["recent"][-25:]): #newest first, max 25 lines
            if line.startswith("[") and "] " in line:
                ts, _, rest = line[1:].partition("] ") #timestamp split
                html.append(f"<div class='ev'><span class='ev-t'>{ts}</span>{rest}</div>")
            else:
                html.append(f"<div class='ev'>{line}</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

#'''Page: Overview of results and best models'''
def page_overview():
    if df_exp.empty:
        sec("Overview")
        empty()
        return
    sec("Headline", "Best model per dataset, measured on the held-out test split")
    total_time = None
    if meta and meta.get("phases"):
        vals = [v.get("elapsed_s") or 0 for v in meta["phases"].values()]
        total_time = sum(vals) or None
    k = st.columns(4)
    kpi(k[0], "Experiments", f"{len(df_exp):,}", f"{df_exp.dataset.nunique()} datasets")
    kpi(k[1], "Models", f"{df_exp.model.nunique()}", f"{df_exp.family.nunique()} families")
    kpi(k[2], "Best F1 macro", f"{df_exp.f1_macro.max():.4f}", df_exp.loc[df_exp.f1_macro.idxmax(), "model"])
    kpi(k[3], "Pipeline time", fmt_dur(total_time) if total_time else "—", "all phases")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    #best model per dataset and task, sorted by F1 macro
    best = (df_exp.sort_values("f1_macro", ascending=False).groupby(["dataset", "task"], as_index=False).first())
    b = best.sort_values("f1_macro")
    f = px.bar(b, x="f1_macro", y="dataset", color="task", orientation="h",barmode="group", text="model")
    f.update_traces(textposition="inside", insidetextanchor="start",textfont=dict(size=10, color="#ffffff")) #model name inside bar
    f.update_xaxes(range=[0, 1.02], title="F1 macro")
    f.update_yaxes(title="")
    chart(f, height=110 + 46 * b.dataset.nunique())

    st.dataframe(best[["dataset", "task", "model", "accuracy", "f1_macro", "mcc","train_time_s"]],width="stretch", hide_index=True, column_config=NUMCOLS)
    sec("Model families", "Does a whole class of algorithms behave differently on this problem?")
    hint("Each point is one experiment. The box shows the middle half of the " "results, the line inside it the median.")
    mc = df_exp[(df_exp.task == "multiclass") & df_exp.family.notna()].copy()
    if mc.empty:
        empty()
    else:
        mc["Family"] = mc.family.map(lambda x: FAMILY_LABEL.get(x, x))
        order = mc.groupby("Family").f1_macro.median().sort_values().index #order by median 
        f = px.box(mc, x="f1_macro", y="Family", points="all", category_orders={"Family": list(order)}, color_discrete_sequence=[PALETTE[0]])
        f.update_traces(marker=dict(size=5, opacity=.55), line=dict(width=1.4)) #all experiments as points
        f.update_xaxes(title="F1 macro", range=[0, 1.02])
        f.update_yaxes(title="")
        chart(f, height=90 + 40 * mc.Family.nunique(), legend=False)
    sec("Hardest attack classes", "Where the detector actually fails — recall per class, across all datasets")
    hint("Recall is the share of real attacks of that class that were caught. " "Low recall means the class slips through.")
    #collect all per-class records (multiclass task)
    rows = []
    for r in exp:
        if r.get("task") != "multiclass":
            continue
        for pc in r.get("per_class", []):
            if pc.get("support", 0) > 0: #class with no samples
                rows.append({"Class": pc["class"], "Dataset": r["dataset"],"Model": r["model"], "Recall": pc["recall"],"Precision": pc["precision"], "Support": pc["support"]})
    if not rows:
        empty()
    else:
        pc_df = pd.DataFrame(rows)
        agg = (pc_df.groupby("Class").agg(Datasets=("Dataset", "nunique"), Rows=("Support", "max"),Worst=("Recall", "min"), Median=("Recall", "median")).reset_index().sort_values("Median"))
        #bar showing median per class, with error bar to worst case
        f = px.bar(agg, x="Median", y="Class", orientation="h",error_x=agg.Median - agg.Worst, text="Median")
        f.update_traces(marker_color=PALETTE[0], texttemplate="%{text:.3f}",textposition="outside", cliponaxis=False,error_x_color=T["ink3"])
        f.update_xaxes(range=[0, 1.12], title="median recall (bar left to worst case)")
        f.update_yaxes(title="")
        chart(f, height=80 + 30 * len(agg), legend=False)
        st.dataframe(agg, width="stretch", hide_index=True, column_config={"Worst": st.column_config.NumberColumn("Worst recall", format="%.4f"),
            "Median": st.column_config.ProgressColumn("Median recall", format="%.4f",min_value=0, max_value=1),
            "Rows": st.column_config.NumberColumn("Rows (max)", format="localized")})

#'''Page: All experiments and models'''
def page_models():
    if df_exp.empty:
        sec("Models")
        empty()
        return

    sec("All experiments", f"{len(df_exp):,} runs — filter, sort, and inspect")
    c1, c2, c3 = st.columns([1, 1, 2])
    ds = c1.multiselect("Dataset", sorted(df_exp.dataset.unique()), placeholder="all")
    tk = c2.multiselect("Task", sorted(df_exp.task.unique()), placeholder="all")
    d = df_exp
    if ds:
        d = d[d.dataset.isin(ds)] #empty selection means all
    if tk:
        d = d[d.task.isin(tk)]
    c3.markdown(f"<div style='padding-top:1.75rem;color:var(--ink-3);font-size:.79rem'>"f"showing {len(d):,} of {len(df_exp):,}</div>", unsafe_allow_html=True)
    st.dataframe(d.drop(columns=["family"]).sort_values("f1_macro", ascending=False),width="stretch", height=400, hide_index=True, column_config=NUMCOLS)
    sec("Model against dataset", "F1 macro, multiclass task")
    mc = df_exp[df_exp.task == "multiclass"]
    if not mc.empty:
        piv = mc.pivot_table(index="model", columns="dataset", values="f1_macro")
        piv = piv.reindex(piv.mean(axis=1).sort_values().index) #best models at the top
        f = px.imshow(piv, color_continuous_scale=T["scale"], aspect="auto",text_auto=".3f", zmin=0, zmax=1)
        f.update_layout(coloraxis_colorbar=dict(title="F1", thickness=10, len=.75))
        f.update_xaxes(title="", side="top")
        f.update_yaxes(title="")
        chart(f, height=110 + 26 * len(piv), legend=False)
    sec("Confusion matrix", "Rebuilt live from the stored matrix — no image files")
    hint("Rows are the true class, columns what the model predicted. " "The diagonal is correct. Shading is row-normalised, so each row sums to 100%.")
    #chained filters to select a single dataset, task, and model (order by F1 macro)
    s1, s2, s3 = st.columns(3)
    ds_pick = s1.selectbox("Dataset", sorted({r["dataset"] for r in exp}))
    pool = [r for r in exp if r["dataset"] == ds_pick]
    tk_pick = s2.selectbox("Task", sorted({r["task"] for r in pool}))
    pool = [r for r in pool if r["task"] == tk_pick]
    pool.sort(key=lambda r: -r.get("f1_macro", 0))
    md_pick = s3.selectbox("Model", [r["model"] for r in pool],format_func=lambda m: m)
    rec = next((r for r in pool if r["model"] == md_pick), None)

    if not rec or not rec.get("confusion_matrix"):
        empty("No stored confusion matrix for this combination.")
    else:
        cm = pd.DataFrame(rec["confusion_matrix"], index=rec["labels"],columns=rec["labels"]).astype(float)
        norm = cm.div(cm.sum(axis=1).clip(lower=1), axis=0) #row-normalised
        left, right = st.columns([3, 2])
        with left:
            f = px.imshow(norm, color_continuous_scale=T["scale"], zmin=0, zmax=1,aspect="auto") 
            f.update_traces(text=cm.astype(int).values, texttemplate="%{text:,}",hovertemplate="true %{y} · predicted %{x}<br>" "%{text:,} flows (%{z:.1%})<extra></extra>")
            f.update_layout(coloraxis_colorbar=dict(title="row %", thickness=10, len=.75))
            f.update_xaxes(title="predicted", side="top")
            f.update_yaxes(title="true")
            chart(f, height=140 + 40 * len(cm), legend=False)
        with right:
            k = st.columns(2)
            kpi(k[0], "F1 macro", f"{rec['f1_macro']:.4f}")
            kpi(k[1], "Accuracy", f"{rec['accuracy']:.4f}")
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            pcd = pd.DataFrame(rec["per_class"])[["class", "support", "precision", "recall", "f1"]]
            st.dataframe(pcd.sort_values("support", ascending=False), width="stretch", hide_index=True, height=140 + 34 * len(cm), column_config={
                             "class": st.column_config.TextColumn("Class"),"support": st.column_config.NumberColumn("Rows", format="localized"),
                             "precision": st.column_config.NumberColumn("Precision", format="%.3f"),"recall": st.column_config.ProgressColumn(
                                "Recall", format="%.3f", min_value=0, max_value=1),"f1": st.column_config.NumberColumn("F1", format="%.3f")})

#'''Page: Hyperparameter tuning results'''
def page_tuning():
    tun = load("tuning_results.json")
    sec("Hyperparameter search","Optuna, 25 trials with 3-fold cross-validation per combination")
    if not tun:
        empty("tuning_results.json not found — the tuning phase has not run.")
        return

    df = pd.DataFrame([{"dataset": r["dataset"], "task": r["task"], "model": r["model"],"default": r["default"]["f1_macro"], "tuned": r["tuned"]["f1_macro"],
        "cv_best": r["best_cv_f1_macro"], "minutes": (r.get("tune_time_s") or 0) / 60} for r in tun])
    df["gain"] = df.tuned - df["default"] #gain from default to tuned
    df["combo"] = df.dataset + " · " + df.task.str[:4] + " · " + df.model

    k = st.columns(4)
    kpi(k[0], "Combinations", f"{len(df)}")
    kpi(k[1], "Improved", f"{(df.gain > 0.001).sum()} of {len(df)}") #threshold to avoid floating point noise
    kpi(k[2], "Largest gain", f"+{df.gain.max():.3f}", df.loc[df.gain.idxmax(), "combo"])
    kpi(k[3], "Search time", fmt_dur(df.minutes.sum() * 60), "all combinations")

    sec("Default against tuned", "Each line is one combination; longer means bigger effect")
    hint("The left dot is the model with the library defaults, the right dot after " "the search. A line pointing left means tuning made it worse.")
    #one line per combination and two marker series
    d = df.sort_values("gain")
    f = go.Figure()
    for _, r in d.iterrows():
        f.add_trace(go.Scatter(x=[r["default"], r.tuned], y=[r.combo, r.combo], mode="lines", line=dict(color=T["ok"] if r.gain >= 0 else T["bad"], width=2),
            showlegend=False, hoverinfo="skip")) #connecting line, no legend, no hover
    f.add_trace(go.Scatter(x=d["default"], y=d.combo, mode="markers", name="default", marker=dict(size=8, color=T["ink3"])))
    f.add_trace(go.Scatter(x=d.tuned, y=d.combo, mode="markers", name="tuned", marker=dict(size=8, color=PALETTE[0])))
    f.update_xaxes(title="F1 macro", range=[0, 1.03])
    f.update_yaxes(title="")
    chart(f, height=140 + 22 * len(d))

    sec("Search history", "How the best score improved over the 25 trials")
    c1, c2 = st.columns([1, 3])
    combo = c1.selectbox("Combination", df.sort_values("gain", ascending=False).combo)
    rec = tun[list(df.combo).index(combo)] #order preserved from df
    tr = pd.DataFrame(rec["trials"])
    if tr.empty:
        empty("No trial history stored.")
    else:
        tr["best"] = tr.value.cummax() #best so far
        f = go.Figure()
        f.add_trace(go.Scatter(x=tr.number, y=tr.value, mode="markers", name="trial",marker=dict(size=7, color=T["ink3"], opacity=.75)))
        f.add_trace(go.Scatter(x=tr.number, y=tr.best, mode="lines", name="best so far",line=dict(color=PALETTE[0], width=2.2)))
        f.update_xaxes(title="trial")
        f.update_yaxes(title="cross-validated F1 macro")
        with c2:
            chart(f, height=300)
        with st.expander("Winning hyperparameters"):
            st.dataframe(pd.DataFrame(rec["best_params"].items(),columns=["Parameter", "Value"]),width="stretch", hide_index=True)

#'''Page: Ablation study results'''
def page_ablations():
    ext = load("extended_results.json")
    sec("Ablation study","What actually moves the needle: imbalance handling, feature selection, ""training size, and how stable the scores are")
    if not ext:
        empty("extended_results.json not found — the ablation phase has not run.")
        return

    df = pd.DataFrame(ext)
    c1, c2 = st.columns(2)
    ds = c1.selectbox("Dataset", sorted(df.dataset.unique()))
    tk = c2.selectbox("Task", sorted(df.task.unique()))
    d = df[(df.dataset == ds) & (df.task == tk)] #kind seperates 4 experiments

    t1, t2, t3, t4 = st.tabs(["Imbalance", "Feature selection","Training size", "Stability"])

    with t1:
        a = d[d.kind == "imbalance"]
        hint("Attack classes are wildly unequal. These are five ways of dealing ""with that, each trained from scratch.")
        if a.empty:
            empty()
        else:
            a = a.copy()
            a["Strategy"] = a.variant.map(lambda v: IMB_LABEL.get(v, v))
            f = px.bar(a, x="model", y="f1_macro", color="Strategy", barmode="group")
            f.update_xaxes(title="")
            f.update_yaxes(title="F1 macro", range=[0, 1.02])
            chart(f, height=420)

    with t2:
        b = d[d.kind == "feature_selection"]
        hint("Fewer features means a smaller, faster model. The question is how ""much accuracy that costs.")
        if b.empty:
            empty()
        else:
            b = b.copy()
            b["Method"] = b.variant.map(lambda v: FS_LABEL.get(v, v))
            #4 dimensions in a chart: x=features kept, y=F1 macro, color=model, size=train time
            f = px.scatter(b, x="n_features", y="f1_macro", color="model",symbol="Method", size="train_time_s", size_max=17)
            f.update_xaxes(title="features kept")
            f.update_yaxes(title="F1 macro")
            chart(f, height=420)
            st.dataframe(b[["model", "Method", "n_features", "f1_macro", "train_time_s"]].sort_values(["model", "n_features"]),width="stretch", hide_index=True, column_config=NUMCOLS)

    with t3:
        c = d[d.kind == "learning_curve"]
        hint("Does more training data still help, or has the model already ""learned everything it can?")
        if c.empty:
            empty()
        else:
            c = c.sort_values("sample_size")
            f = px.line(c, x="sample_size", y="f1_macro", color="model", markers=True)
            f.update_xaxes(type="log", title="training rows (log scale)")
            f.update_yaxes(title="F1 macro")
            chart(f, height=420)

    with t4:
        e = d[d.kind == "cv_fold"] #individual folds (visible variablity)
        hint("Five folds of the same data. A tall box means the score depends on ""which rows happened to land in the test split.")
        if e.empty:
            empty()
        else:
            order = e.groupby("model").f1_macro.median().sort_values().index
            f = px.box(e, x="model", y="f1_macro", points="all",category_orders={"model": list(order)},color_discrete_sequence=[PALETTE[0]])
            f.update_traces(marker=dict(size=6, opacity=.7), line=dict(width=1.4))
            f.update_xaxes(title="")
            f.update_yaxes(title="F1 macro")
            chart(f, height=420, legend=False)
            summ = (e.groupby("model").f1_macro.agg(["mean", "std", "min", "max"]).reset_index().sort_values("mean", ascending=False))
            st.dataframe(summ, width="stretch", hide_index=True, column_config={"model": st.column_config.TextColumn("Model"),
                "mean": st.column_config.NumberColumn("Mean", format="%.4f"),"std": st.column_config.NumberColumn("Std dev", format="%.4f"),
                "min": st.column_config.NumberColumn("Min", format="%.4f"),"max": st.column_config.NumberColumn("Max", format="%.4f")})

#'''Page: Leave-one-dataset-out generalisation results'''
def page_generalisation():
    sec("Leave-one-dataset-out","Train on three networks, test on the fourth — never seen during training")
    hint("This is the honest test: an IDS installed on a network it was not ""trained on. Compare these numbers with the ones on the Overview page.")

    lodo = [r for r in results if str(r.get("dataset", "")).startswith("LODO")]
    if not lodo:
        empty("No cross-dataset results.")
    else:
        d = pd.DataFrame([{"Held-out network": r["dataset"].replace("LODO_holdout_", ""),"F1 macro": r["f1_macro"], "Accuracy": r["accuracy"],
                           "MCC": r.get("mcc"), "Test rows": r.get("test_rows")}for r in lodo]).sort_values("F1 macro")
        #comparison of unseen network against the best model trained on that network
        in_dom = (df_exp[df_exp.task == "multiclass"].groupby("dataset").f1_macro.max()) if not df_exp.empty else {}
        d["Same-network best"] = d["Held-out network"].map(in_dom)

        f = go.Figure()
        f.add_trace(go.Bar(x=d["F1 macro"], y=d["Held-out network"], orientation="h",name="unseen network", marker_color=PALETTE[0]))
        if d["Same-network best"].notna().any():
            f.add_trace(go.Scatter(x=d["Same-network best"], y=d["Held-out network"],mode="markers", name="best on its own network",marker=dict(size=10, color=T["ink3"], symbol="diamond")))
        f.update_xaxes(title="F1 macro", range=[0, 1.02])
        f.update_yaxes(title="")
        chart(f, height=150 + 44 * len(d))
        st.dataframe(d, width="stretch", hide_index=True, column_config={"F1 macro": st.column_config.ProgressColumn("F1 macro", format="%.4f",min_value=0, max_value=1),
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),"MCC": st.column_config.NumberColumn(format="%.4f"),
            "Same-network best": st.column_config.NumberColumn(format="%.4f"),"Test rows": st.column_config.NumberColumn(format="localized")})
    ad = load("adaptation_results.json")
    sec("How much local data closes the gap","Add a slice of the target network's own traffic and retrain")
    hint("Zero percent is pure leave-one-out. The curve shows how quickly the " "detector recovers once it sees a little traffic from the new network.")
    if not ad:
        empty("adaptation_results.json not found.")
        return
    a = pd.DataFrame(ad)
    a["percent"] = a.fraction * 100
    f = px.line(a.sort_values("percent"), x="percent", y="f1_macro",color="holdout", markers=True)
    f.update_xaxes(title="local traffic used for adaptation (%)")
    f.update_yaxes(title="F1 macro on the target network", range=[0, 1.02])
    chart(f, height=430)

    piv = a.pivot_table(index="holdout", columns="percent", values="f1_macro")
    piv.columns = [f"{c:g}%" for c in piv.columns] 
    st.dataframe(piv.round(4), width="stretch")

#'''Page: Files generated by the pipeline'''
def page_files():
    sec("Figures", "PNG files written by the pipeline — these are the paper assets")
    if not fig_dir.exists():
        empty("No figures directory.")
    else:
        pngs = sorted(fig_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) #newest first
        if not pngs:
            empty("No figures yet.")
        else:
            groups = sorted({p.stem.split("_")[0] for p in pngs}) #grouped by the filename
            c1, c2 = st.columns([1, 3])
            pick = c1.selectbox("Group", ["all"] + groups)
            shown = pngs if pick == "all" else [p for p in pngs if p.stem.startswith(pick)]
            c2.markdown(f"<div style='padding-top:1.75rem;color:var(--ink-3);" f"font-size:.79rem'>{len(shown)} files, showing first 12</div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, p in enumerate(shown[:12]): #limit of 12 images to avoid overloading the browser
                with cols[i % 3]:
                    try:
                        st.image(str(p), caption=p.name, width="stretch")
                    except Exception:
                        note(f"{p.name} — being written", "warn") #first is being written now

    sec("Logs")
    logs = sorted(res_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        empty("No log files.")
        return
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("File", logs, format_func=lambda p: p.name, label_visibility="collapsed")
    n = c2.number_input("Lines", 20, 500, 60, step=20, label_visibility="collapsed")
    try:
        lines = choice.read_text(errors="ignore").splitlines()
        st.code("\n".join(lines[-int(n):]) or "(empty)", language=None) #only last n lines
        st.markdown(f"<div class='sec-s'>{len(lines):,} lines · modified " f"{fmt_dur(time.time() - choice.stat().st_mtime)} ago</div>", unsafe_allow_html=True)
    except Exception as e:
        note(f"Could not read the file: {e}", "bad")

#dictionary for function call
if not prog and not results and page != "Files":
    sec("No data")
    note("Nothing recorded yet. Start with <span class='mono'>python src/runfull.py</span>.", "warn")
else:
    {"Run": page_run, "Overview": page_overview, "Models": page_models, "Tuning": page_tuning, "Ablations": page_ablations, "Generalisation": page_generalisation, "Files": page_files}[page]()

#automatic refresh if the pipeline is still running an active phase
pipeline_active = bool(meta and meta.get("status") not in ("done", "failed"))
phase_active = bool(prog and prog.get("stage") != "finished")
if (auto_refresh and view == "current" and page == "Run" and (pipeline_active or phase_active)):
    time.sleep(refresh_sec)
    st.rerun()
