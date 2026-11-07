"""
Train five classifiers, evaluate them on the held-out test set, and save everything.

Run from the project root:
    python -m src.train

Output (all in model/):
    <model>.joblib   -> one full pipeline per classifier (preprocessing + model)
    metrics.csv      -> comparison table of the six metrics
    metrics.md       -> the same table in Markdown, ready for the README
    model_info.json  -> training metadata (versions, features, data sizes)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import compute_metrics, get_scores
from src.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

TRAIN_FILE = Path("data/processed/train.csv")
TEST_FILE = Path("test_data.csv")
MODEL_DIR = Path("model")
RANDOM_STATE = 42

# Display name -> saved file. The Streamlit app reads this same mapping.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_preprocessor():
    """Impute, scale and encode: fitted on training data only, saved inside each pipeline."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    boolean = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("bool", boolean, BOOLEAN_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_models():
    """The five required classifiers with sensible, documented settings."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def to_markdown(table):
    """Render the metrics table as Markdown without extra dependencies."""
    header = "| ML Model Name | " + " | ".join(table.columns) + " |"
    divider = "|---" * (len(table.columns) + 1) + "|"
    rows = [
        f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |"
        for name, row in table.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def main():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    MODEL_DIR.mkdir(exist_ok=True)
    results = []

    for name, classifier in build_models().items():
        pipeline = Pipeline([("prep", build_preprocessor()), ("model", classifier)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = get_scores(pipeline, X_test)
        results.append({"Model": name, **compute_metrics(y_test, y_pred, y_score)})

        joblib.dump(pipeline, MODEL_DIR / MODEL_FILES[name])
        print(f"Trained and saved: {name}")

    table = pd.DataFrame(results).set_index("Model").round(4)
    table.to_csv(MODEL_DIR / "metrics.csv")
    (MODEL_DIR / "metrics.md").write_text(to_markdown(table), encoding="utf-8")

    info = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scikit_learn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "n_train": len(train),
        "n_test": len(test),
        "features": FEATURES,
        "target": TARGET,
        "models": MODEL_FILES,
        "train_goal_rate": round(float(y_train.mean()), 4),
    }
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\n=== Test-set performance ===")
    print(table.to_string())
    print(f"\nSaved models and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()