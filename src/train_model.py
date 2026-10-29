"""
Adaptive IoV Risk Classifier
-----------------------------
Trains a model to classify each network telemetry record into
{normal, medium, emergency}. This risk level then drives the ADAPTIVE
security-selection logic (see src/adaptive_security.py):
    normal    -> ASCON only (lightweight AEAD, minimum latency)
    medium    -> ASCON + ML-KEM (post-quantum key establishment)
    emergency -> ASCON + ML-KEM + ML-DSA (full PQC: key exchange + signature)

Outputs (into reports/):
    model.pkl                  trained pipeline (joblib)
    confusion_matrix.png
    feature_importance.png
    classification_report.txt
    metrics_summary.json
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "iov_security_dataset.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

FEATURES = [
    "speed_kmh", "message_priority", "packet_size_bytes", "latency_ms",
    "packet_loss_pct", "network_load_pct", "distance_m",
    "message_frequency_hz", "auth_failures", "anomaly_score",
]
TARGET = "risk_level"
CLASS_ORDER = ["normal", "medium", "emergency"]


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create sample weights: boost emergency class HEAVILY
    sample_weights = y_train.map({
        "normal": 1.0,
        "medium": 2.0,
        "emergency": 10.0,  # 10x weight for emergency cases
    })

    # Custom class weights to extremely prioritize emergency recall
    class_weights = {
        "normal": 1.0,
        "medium": 2.0,
        "emergency": 8.0,  # 8x weight for emergency detection
    }
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=800, max_depth=10, min_samples_leaf=1,
            class_weight=class_weights, random_state=42, n_jobs=-1,
            min_samples_split=3, criterion="gini",
        )),
    ])

    pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro", n_jobs=-1)

    report_txt = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, target_names=CLASS_ORDER
    )
    print(report_txt)
    print(f"Accuracy: {acc:.4f}  F1-macro: {f1_macro:.4f}")
    print(f"5-fold CV F1-macro: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    (REPORTS / "classification_report.txt").write_text(report_txt)

    metrics = dict(
        accuracy=acc, f1_macro=f1_macro, precision_macro=prec_macro,
        recall_macro=rec_macro, cv_f1_macro_mean=float(cv_scores.mean()),
        cv_f1_macro_std=float(cv_scores.std()),
        n_train=len(X_train), n_test=len(X_test),
    )
    (REPORTS / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=CLASS_ORDER, cmap="Blues", ax=ax, colorbar=True
    )
    ax.set_title("Confusion Matrix - IoV Risk Classifier")
    plt.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Feature importance
    importances = pipe.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=importances[order], y=np.array(FEATURES)[order], ax=ax, color="#2b6cb0")
    ax.set_title("Feature Importance - Risk Classification")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(REPORTS / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Class distribution plot
    fig, ax = plt.subplots(figsize=(5, 4))
    df[TARGET].value_counts().reindex(CLASS_ORDER).plot(kind="bar", ax=ax, color=["#2f855a", "#d69e2e", "#c53030"])
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of records")
    plt.tight_layout()
    fig.savefig(REPORTS / "class_distribution.png", dpi=150)
    plt.close(fig)

    joblib.dump(pipe, REPORTS / "model.pkl")
    print(f"\nSaved model + reports to: {REPORTS}")


if __name__ == "__main__":
    main()
