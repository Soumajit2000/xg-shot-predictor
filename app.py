"""
xG Shot Predictor: Streamlit app.

Upload the held-out test CSV, choose one of five trained classifiers, and see how
it performs. Models are loaded from model/; nothing is retrained here.

Run locally:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Rectangle
from sklearn.metrics import classification_report, confusion_matrix

from src.evaluate import compute_metrics, get_scores
from src.features import FEATURES, TARGET
from src.train import MODEL_FILES

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
XG_MODEL = "Logistic Regression"  # overall winner, used for the shot map

# Design tokens
TURF = "#2E6B45"
TURF_DARK = "#285E3D"
PAGE = "#FAFBF8"
INK = "#15211A"
LINE = "#C3CEC6"
PITCH = "#F1F5F1"
MISS = "#8A988F"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})

st.set_page_config(page_title="xG Shot Predictor", page_icon="⚽", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], button, input, label {
    font-family: 'Barlow', system-ui, sans-serif;
}
h1, h2, h3 { font-family: 'Barlow Condensed', 'Barlow', sans-serif; font-weight: 600; }
#MainMenu, footer { visibility: hidden; }

.hero {
    background: repeating-linear-gradient(90deg, #2E6B45 0 56px, #285E3D 56px 112px);
    border-radius: 6px;
    padding: 28px 32px;
    color: #FFFFFF;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    margin-bottom: 8px;
}
.hero h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1;
    margin: 0 0 10px 0;
    padding: 0;
    color: #FFFFFF;
}
.hero p { margin: 0; max-width: 60ch; font-size: 1.05rem; line-height: 1.45; opacity: 0.92; }
.hero svg { flex-shrink: 0; }
@media (max-width: 760px) { .hero svg { display: none; } .hero h1 { font-size: 2.2rem; } }

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D8E0DA;
    border-left: 4px solid #2E6B45;
    border-radius: 4px;
    padding: 10px 14px;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 600;
}
button[data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
</style>
"""

HERO = f"""
<div class="hero">
  <div>
    <h1>xG Shot Predictor</h1>
    <p>How likely was that shot to go in? Five classifiers trained on 5,600 shots
    from the 2018 and 2022 World Cups and Euro 2020 and 2024, using where the shot
    was taken, how it was struck and where the defenders stood.</p>
  </div>
  <svg width="170" height="96" viewBox="0 0 170 96" fill="none" stroke="#FFFFFF"
       stroke-opacity="0.85" stroke-width="2" aria-hidden="true">
    <rect x="61" y="2" width="48" height="5"/>
    <rect x="15" y="7" width="140" height="56"/>
    <rect x="50" y="7" width="70" height="22"/>
    <path d="M62 63 A 26 26 0 0 0 108 63"/>
    <circle cx="85" cy="47" r="2" fill="#FFFFFF"/>
  </svg>
</div>
"""

st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading and evaluation
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all five saved pipelines once and keep them in memory."""
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


@st.cache_data
def load_model_info():
    return json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))


def evaluate(model, df):
    X, y_true = df[FEATURES], df[TARGET]
    y_pred = model.predict(X)
    metrics = compute_metrics(y_true, y_pred, get_scores(model, X))
    return y_true, y_pred, metrics


def to_true_xg(p, base_rate):
    """Undo class_weight='balanced' so probabilities reflect the real goal rate."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * base_rate / (1 - base_rate)
    return odds / (1 + odds)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No goal", "Goal"]
    cmap = LinearSegmentedColormap.from_list("turf", ["#FFFFFF", TURF])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.patch.set_facecolor(PAGE)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, linewidths=2,
                linecolor=PAGE, xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def draw_pitch(bottom=70):
    """Attacking end of a StatsBomb pitch, drawn vertically with the goal at the top."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(PAGE)
    line = {"edgecolor": LINE, "linewidth": 1.2, "fill": False}

    ax.add_patch(Rectangle((0, 60), 80, 60, facecolor=PITCH, edgecolor=LINE, linewidth=1.2))
    ax.add_patch(Rectangle((18, 102), 44, 18, **line))      # penalty area
    ax.add_patch(Rectangle((30, 114), 20, 6, **line))       # six-yard box
    ax.add_patch(Rectangle((36, 120), 8, 1.8, edgecolor=INK, linewidth=1.4, fill=False))  # goal
    ax.add_patch(Arc((40, 108), 20, 20, theta1=217, theta2=323, color=LINE, lw=1.2))
    ax.add_patch(Arc((40, 60), 20, 20, theta1=0, theta2=180, color=LINE, lw=1.2))
    ax.scatter(40, 108, s=6, color=LINE)

    ax.set_xlim(-1, 81)
    ax.set_ylim(bottom, 123)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def marker_size(xg):
    return 12 + 650 * np.asarray(xg)


def plot_shot_map(df, xg):
    fig, ax = draw_pitch(bottom=min(70, df["x"].min() - 2))
    across, up = df["y"].to_numpy(), df["x"].to_numpy()
    goals = df[TARGET].to_numpy() == 1

    ax.scatter(across[~goals], up[~goals], s=marker_size(xg[~goals]),
               facecolors="none", edgecolors=MISS, linewidths=0.8, alpha=0.6)
    ax.scatter(across[goals], up[goals], s=marker_size(xg[goals]),
               facecolors=TURF, edgecolors="#FFFFFF", linewidths=0.7, alpha=0.95, zorder=3)

    handles = [
        ax.scatter([], [], s=90, facecolors=TURF, edgecolors="#FFFFFF", label="Goal"),
        ax.scatter([], [], s=90, facecolors="none", edgecolors=MISS, label="No goal"),
    ] + [
        ax.scatter([], [], s=marker_size(v), facecolors="none", edgecolors=INK,
                   linewidths=0.6, label=f"{v:.2f} xG")
        for v in (0.05, 0.2, 0.5)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
              ncol=5, frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Upload test data")
uploaded = st.sidebar.file_uploader("CSV with the 18 features and is_goal", type="csv")
st.sidebar.download_button(
    "Download sample test_data.csv",
    data=SAMPLE_CSV.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)

st.sidebar.header("Choose a model")
model_name = st.sidebar.selectbox("Classifier", list(MODEL_FILES))

st.sidebar.divider()
st.sidebar.caption(
    "Data from [StatsBomb Open Data](https://github.com/statsbomb/open-data). "
    "Models were trained once, offline; this app only loads and evaluates them."
)

# ---------------------------------------------------------------------------
# Header and empty state
# ---------------------------------------------------------------------------
st.markdown(HERO, unsafe_allow_html=True)

if uploaded is None:
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Start with a test file")
        st.write(
            "Upload a CSV of shots from the sidebar. Each row needs the 18 shot "
            "features plus `is_goal`. No file handy? Download the sample first, "
            "then upload it."
        )
        with st.expander("See the required columns"):
            st.write(", ".join(f"`{c}`" for c in FEATURES + [TARGET]))
    with right:
        st.pyplot(draw_pitch()[0])
    st.stop()

# ---------------------------------------------------------------------------
# Validate upload
# ---------------------------------------------------------------------------
df = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"This file is missing required columns: {', '.join(missing)}. "
             "Add them, or download the sample file to see the expected format.")
    st.stop()
if df[TARGET].nunique() < 2:
    st.error("The is_goal column needs both goals (1) and non-goals (0) to compute the metrics.")
    st.stop()

models = load_models()
tab_perf, tab_compare, tab_map = st.tabs(["Performance", "Compare models", "Shot map"])

# ---------------------------------------------------------------------------
# Tab 1: selected model
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader(model_name)
    y_true, y_pred, metrics = evaluate(models[model_name], df)

    for col, (name, value) in zip(st.columns(6), metrics.items()):
        col.metric(name, f"{value:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    st.write(
        f"In football terms: the model caught **{tp} of {tp + fn} goals**. "
        f"Of the **{tp + fp} shots** it called as goals, **{tp}** went in."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion_matrix(y_true, y_pred))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, y_pred, target_names=["No goal", "Goal"], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: all models side by side
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("All five models on this file")
    table = pd.DataFrame({name: evaluate(m, df)[2] for name, m in models.items()}).T
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#D5E8DA"),
        use_container_width=True,
    )
    st.bar_chart(table[["AUC", "F1", "MCC"]], color=[TURF, "#7FB394", "#C3CEC6"])
    st.caption(
        f"Accuracy flatters here: always predicting no goal would already score "
        f"{1 - df[TARGET].mean():.1%} while catching zero goals. "
        "AUC, F1 and MCC show which models actually find goals."
    )

# ---------------------------------------------------------------------------
# Tab 3: shot map
# ---------------------------------------------------------------------------
with tab_map:
    base_rate = load_model_info().get("train_goal_rate", 0.091)
    raw = models[XG_MODEL].predict_proba(df[FEATURES])[:, 1]
    xg = to_true_xg(raw, base_rate)

    stats, pitch = st.columns([1, 3])
    with stats:
        st.subheader("Every shot")
        st.metric("Shots", f"{len(df):,}")
        st.metric("Goals", int(df[TARGET].sum()))
        st.metric("Expected goals", f"{xg.sum():.1f}")
        st.caption(
            "Circle size is the chance quality from the Logistic Regression model, "
            "the overall winner. Filled circles went in. When expected goals sits "
            "close to actual goals, the model's probabilities are well calibrated."
        )
    with pitch:
        st.pyplot(plot_shot_map(df, xg))