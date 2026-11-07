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
from matplotlib.patches import Arc, Rectangle
from sklearn.metrics import classification_report, confusion_matrix

from src.evaluate import compute_metrics, get_scores
from src.features import FEATURES, TARGET
from src.train import MODEL_FILES

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
XG_MODEL = "Logistic Regression"  # overall winner, used for the shot map

st.set_page_config(page_title="xG Shot Predictor", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Loading
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
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def draw_half_pitch():
    """Attacking half of a StatsBomb pitch (120 x 80 units)."""
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_facecolor("#3a7d44")
    fig.patch.set_facecolor("#3a7d44")
    line = {"edgecolor": "white", "linewidth": 1.5, "fill": False}

    ax.add_patch(Rectangle((60, 0), 60, 80, **line))      # half pitch
    ax.add_patch(Rectangle((102, 18), 18, 44, **line))    # penalty area
    ax.add_patch(Rectangle((114, 30), 6, 20, **line))     # six-yard box
    ax.add_patch(Arc((108, 40), 20, 20, theta1=127, theta2=233, color="white", lw=1.5))
    ax.add_patch(Arc((60, 40), 20, 20, theta1=270, theta2=90, color="white", lw=1.5))
    ax.plot([120, 120], [36, 44], color="white", lw=5)     # goal mouth
    ax.scatter(108, 40, s=12, color="white")               # penalty spot

    ax.set_xlim(58, 122)
    ax.set_ylim(-2, 82)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def plot_shot_map(df, xg):
    fig, ax = draw_half_pitch()
    goals = df[TARGET].to_numpy() == 1
    order = np.argsort(xg)  # draw dangerous shots on top
    points = ax.scatter(
        df["x"].to_numpy()[order], df["y"].to_numpy()[order],
        c=xg[order], s=20 + 400 * xg[order], cmap="YlOrRd", vmin=0, vmax=0.6,
        edgecolors=np.where(goals[order], "black", "none"), linewidths=1.2, alpha=0.85,
    )
    bar = fig.colorbar(points, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Expected goals (xG)", color="white")
    bar.ax.tick_params(colors="white")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("1 · Upload test data")
uploaded = st.sidebar.file_uploader("Test CSV with the 18 features and is_goal", type="csv")
st.sidebar.download_button(
    "⬇️ Download sample test_data.csv",
    data=SAMPLE_CSV.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)

st.sidebar.header("2 · Choose a model")
model_name = st.sidebar.selectbox("Classifier", list(MODEL_FILES))

st.sidebar.divider()
st.sidebar.caption(
    "Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data). "
    "Models trained once offline; this app only loads and evaluates them."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚽ xG Shot Predictor")
st.markdown(
    "**Will this shot be a goal?** Five classifiers trained on every shot from the "
    "2018 & 2022 World Cups and Euro 2020 & 2024, using shot geometry, game "
    "situation and defender positions at the moment of the strike."
)

if uploaded is None:
    st.info(
        "👈 Upload a test CSV from the sidebar to begin. "
        "No file handy? Download the sample `test_data.csv` from the sidebar first."
    )
    with st.expander("Which columns does the CSV need?"):
        st.write(", ".join(f"`{c}`" for c in FEATURES + [TARGET]))
    st.stop()

# ---------------------------------------------------------------------------
# Validate upload
# ---------------------------------------------------------------------------
df = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()
if df[TARGET].nunique() < 2:
    st.error("The target column needs both goals (1) and non-goals (0) to compute the metrics.")
    st.stop()

st.success(
    f"Loaded **{len(df):,} shots**, of which **{int(df[TARGET].sum())} were goals** "
    f"({df[TARGET].mean():.1%})."
)

models = load_models()
tab_perf, tab_compare, tab_map = st.tabs(
    ["📊 Model performance", "🏆 Compare all models", "🗺️ Shot map"]
)

# ---------------------------------------------------------------------------
# Tab 1: selected model
# ---------------------------------------------------------------------------
with tab_perf:
    st.subheader(model_name)
    y_true, y_pred, metrics = evaluate(models[model_name], df)

    for col, (name, value) in zip(st.columns(6), metrics.items()):
        col.metric(name, f"{value:.3f}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    st.caption(
        f"In plain football terms: the model caught **{tp} of {tp + fn} goals**, "
        f"and of the **{tp + fp} shots** it flagged as goals, **{tp}** actually went in."
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
    st.subheader("All five models on the uploaded data")
    table = pd.DataFrame({name: evaluate(m, df)[2] for name, m in models.items()}).T
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#a5d6a7"),
        use_container_width=True,
    )
    st.bar_chart(table[["AUC", "F1", "MCC"]])
    st.caption(
        f"Why accuracy misleads here: always predicting 'no goal' would already "
        f"score {1 - df[TARGET].mean():.1%} accuracy while catching zero goals. "
        "MCC, F1 and AUC tell the real story."
    )

# ---------------------------------------------------------------------------
# Tab 3: shot map
# ---------------------------------------------------------------------------
with tab_map:
    st.subheader("Shot map coloured by expected goals")
    base_rate = load_model_info().get("train_goal_rate", 0.091)
    raw = models[XG_MODEL].predict_proba(df[FEATURES])[:, 1]
    xg = to_true_xg(raw, base_rate)

    c1, c2, c3 = st.columns(3)
    c1.metric("Shots", f"{len(df):,}")
    c2.metric("Total xG", f"{xg.sum():.1f}")
    c3.metric("Actual goals", int(df[TARGET].sum()))

    st.pyplot(plot_shot_map(df, xg))
    st.caption(
        f"xG from the {XG_MODEL} model, the overall winner. Bigger, redder circles are "
        "better chances; circles outlined in black were actual goals."
    )