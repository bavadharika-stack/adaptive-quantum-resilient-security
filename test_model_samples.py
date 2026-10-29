"""
Test Suite for Trained IoV Risk Classifier
============================================
Generates a separate sample test dataset and evaluates the trained model.
All results saved to: reports/sample_testresult/
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "reports" / "model.pkl"
TEST_RESULTS_DIR = ROOT / "reports" / "sample_testresult"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "speed_kmh", "message_priority", "packet_size_bytes", "latency_ms",
    "packet_loss_pct", "network_load_pct", "distance_m",
    "message_frequency_hz", "auth_failures", "anomaly_score",
]

def clip(a, lo, hi):
    return np.clip(a, lo, hi)


def generate_sample_test_data(n_samples=200, random_seed=999):
    """
    Generate a small test dataset (separate from training data).
    Uses a different random seed to ensure different samples.
    """
    rng = np.random.default_rng(random_seed)
    
    # Create balanced test set: ~67 normal, ~67 medium, ~66 emergency
    n_normal = n_samples // 3
    n_medium = n_samples // 3
    n_emergency = n_samples - n_normal - n_medium
    
    test_data = []
    labels = []
    
    # NORMAL samples
    for _ in range(n_normal):
        sample = {
            "speed_kmh": clip(rng.normal(60, 18), 0, 140),
            "message_priority": rng.choice([1, 2, 3], p=[0.70, 0.25, 0.05]),
            "packet_size_bytes": clip(rng.normal(300, 40), 100, 500),
            "distance_m": clip(rng.normal(150, 60), 5, 500),
            "network_load_pct": clip(rng.normal(35, 12), 2, 100),
            "message_frequency_hz": clip(rng.normal(10, 1.5), 1, 20),
            "latency_ms": clip(5 + 0.02 * clip(rng.normal(150, 60), 5, 500) + 
                              0.25 * clip(rng.normal(35, 12), 2, 100) + 
                              0.03 * clip(rng.normal(60, 18), 0, 140) + 
                              rng.normal(0, 2.5), 1, 400),
            "packet_loss_pct": clip(0.01 * clip(rng.normal(35, 12), 2, 100) + 
                                   0.01 * (clip(rng.normal(60, 18), 0, 140) / 10) + 
                                   rng.normal(0, 1.0), 0, 15),
            "auth_failures": rng.poisson(0.05),
            "anomaly_score": clip(0.02 * clip(rng.normal(35, 12), 2, 100) + 
                                 0.3 * rng.poisson(0.05) + 
                                 rng.normal(5, 3), 0, 100),
        }
        test_data.append(sample)
        labels.append("normal")
    
    # MEDIUM samples
    for _ in range(n_medium):
        sample = {
            "speed_kmh": clip(rng.normal(75, 25), 0, 160),
            "message_priority": rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25]),
            "packet_size_bytes": clip(rng.normal(340, 60), 100, 800),
            "distance_m": clip(rng.normal(280, 90), 20, 900),
            "network_load_pct": clip(rng.normal(65, 15), 20, 100),
            "message_frequency_hz": clip(rng.normal(8, 3), 0.5, 20),
            "latency_ms": clip(20 + 0.05 * clip(rng.normal(280, 90), 20, 900) + 
                              0.55 * clip(rng.normal(65, 15), 20, 100) + 
                              0.08 * clip(rng.normal(75, 25), 0, 160) + 
                              rng.normal(0, 6), 5, 700),
            "packet_loss_pct": clip(0.06 * clip(rng.normal(65, 15), 20, 100) + 
                                   0.03 * (clip(rng.normal(75, 25), 0, 160) / 10) + 
                                   rng.normal(2, 2.5), 0, 40),
            "auth_failures": rng.poisson(0.8),
            "anomaly_score": clip(0.35 * clip(rng.normal(65, 15), 20, 100) + 
                                 3.0 * rng.poisson(0.8) + 
                                 rng.normal(20, 8), 0, 100),
        }
        test_data.append(sample)
        labels.append("medium")
    
    # EMERGENCY samples (using our improved features)
    for _ in range(n_emergency):
        scenario = rng.choice(["collision", "dos_jam", "spoof_replay"])
        
        if scenario == "collision":
            sample = {
                "speed_kmh": clip(rng.normal(100, 25), 20, 200),
                "message_priority": 3,
                "packet_size_bytes": clip(rng.normal(320, 50), 100, 600),
                "distance_m": clip(rng.normal(15, 8), 1, 60),
                "network_load_pct": clip(rng.normal(55, 15), 10, 100),
                "message_frequency_hz": clip(rng.normal(15, 3), 5, 20),
                "latency_ms": clip(15 + 0.3 * clip(rng.normal(15, 8), 1, 60) + 
                                  rng.normal(5, 4), 1, 200),
                "packet_loss_pct": clip(rng.normal(8, 5), 0, 40),
                "auth_failures": clip(rng.poisson(2.5) + 2, 2, 20),
                "anomaly_score": clip(rng.normal(70, 12), 35, 100),
            }
        elif scenario == "dos_jam":
            sample = {
                "speed_kmh": clip(rng.normal(55, 25), 0, 150),
                "message_priority": 3,
                "packet_size_bytes": clip(rng.normal(500, 150), 100, 1200),
                "distance_m": clip(rng.normal(200, 100), 5, 800),
                "network_load_pct": clip(rng.normal(95, 6), 70, 100),
                "message_frequency_hz": clip(rng.normal(19, 4), 5, 40),
                "latency_ms": clip(rng.normal(350, 120), 50, 900),
                "packet_loss_pct": clip(rng.normal(45, 18), 10, 95),
                "auth_failures": rng.poisson(4.5),
                "anomaly_score": clip(rng.normal(85, 10), 40, 100),
            }
        else:  # spoof_replay
            sample = {
                "speed_kmh": clip(rng.normal(70, 40), 0, 220),
                "message_priority": 3,
                "packet_size_bytes": clip(rng.normal(300, 80), 100, 900),
                "distance_m": clip(rng.normal(120, 90), 1, 900),
                "network_load_pct": clip(rng.normal(60, 20), 10, 100),
                "message_frequency_hz": clip(rng.normal(12, 5), 1, 30),
                "latency_ms": clip(rng.normal(120, 60), 5, 500),
                "packet_loss_pct": clip(rng.normal(20, 12), 0, 70),
                "auth_failures": clip(rng.poisson(6.0) + 3, 3, 25),
                "anomaly_score": clip(rng.normal(90, 8), 60, 100),
            }
        
        test_data.append(sample)
        labels.append("emergency")
    
    df = pd.DataFrame(test_data)
    df["true_label"] = labels
    
    return df


def test_model():
    """Load model and evaluate on sample test data."""
    
    print("\n" + "="*70)
    print("IoV RISK CLASSIFIER - SAMPLE TEST SUITE")
    print("="*70)
    
    # Step 1: Generate sample test data
    print("\n[1/5] Generating sample test dataset (200 samples)...")
    test_df = generate_sample_test_data(n_samples=200, random_seed=999)
    X_test = test_df[FEATURES]
    y_true = test_df["true_label"]
    
    print(f"      ✓ Generated {len(test_df)} test samples")
    print(f"      Class distribution:")
    for label in ["normal", "medium", "emergency"]:
        count = (y_true == label).sum()
        pct = 100 * count / len(y_true)
        print(f"        - {label}: {count} ({pct:.1f}%)")
    
    # Step 2: Load trained model
    print("\n[2/5] Loading trained model...")
    if not MODEL_PATH.exists():
        print(f"      ✗ ERROR: Model not found at {MODEL_PATH}")
        print("      Please run: python src/train_model.py")
        return
    
    model = joblib.load(MODEL_PATH)
    print(f"      ✓ Model loaded successfully")
    print(f"      Model classes: {model.classes_}")
    
    # Step 3: Make predictions
    print("\n[3/5] Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Apply threshold-based classification (same as adaptive_security.py)
    emergency_idx = list(model.classes_).index("emergency")
    y_pred_adjusted = []
    for i, proba in enumerate(y_proba):
        if proba[emergency_idx] > 0.15:
            y_pred_adjusted.append("emergency")
        else:
            y_pred_adjusted.append(y_pred[i])  # Use the standard prediction
    
    print(f"      ✓ Predictions complete")
    print(f"      Using threshold-based emergency detection (prob > 0.15)")
    
    # Step 4: Calculate metrics
    print("\n[4/5] Calculating performance metrics...")
    acc = accuracy_score(y_true, y_pred_adjusted)
    prec_macro = precision_score(y_true, y_pred_adjusted, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred_adjusted, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_adjusted, average="macro", zero_division=0)
    
    print(f"      ✓ Metrics calculated:")
    print(f"        - Accuracy: {acc:.4f}")
    print(f"        - Precision (macro): {prec_macro:.4f}")
    print(f"        - Recall (macro): {rec_macro:.4f}")
    print(f"        - F1-Score (macro): {f1_macro:.4f}")
    
    class_report = classification_report(
        y_true, y_pred_adjusted, 
        labels=["normal", "medium", "emergency"],
        target_names=["normal", "medium", "emergency"],
        zero_division=0
    )
    
    print("\n      Classification Report:")
    print("      " + "\n      ".join(class_report.split("\n")))
    
    # Step 5: Save reports
    print("\n[5/5] Saving detailed reports...")
    
    # Save test dataset
    test_df.to_csv(TEST_RESULTS_DIR / "sample_test_dataset.csv", index=False)
    print(f"      ✓ Test dataset → sample_test_dataset.csv")
    
    # Save predictions
    results_df = test_df.copy()
    results_df["predicted_label"] = y_pred_adjusted
    results_df["correct"] = (results_df["true_label"] == results_df["predicted_label"]).astype(int)
    results_df.to_csv(TEST_RESULTS_DIR / "predictions_with_results.csv", index=False)
    print(f"      ✓ Predictions → predictions_with_results.csv")
    
    # Save classification report
    (TEST_RESULTS_DIR / "classification_report.txt").write_text(class_report)
    print(f"      ✓ Classification report → classification_report.txt")
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_true, y_pred_adjusted, labels=["normal", "medium", "emergency"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["normal", "medium", "emergency"],
                yticklabels=["normal", "medium", "emergency"],
                ax=ax)
    ax.set_title("Confusion Matrix - Sample Test Results")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig(TEST_RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"      ✓ Confusion matrix → confusion_matrix.png")
    
    # Save detailed metrics JSON
    metrics = {
        "test_timestamp": datetime.now().isoformat(),
        "n_samples": len(test_df),
        "class_distribution": {
            "normal": int((y_true == "normal").sum()),
            "medium": int((y_true == "medium").sum()),
            "emergency": int((y_true == "emergency").sum()),
        },
        "performance_metrics": {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
        },
        "per_class_metrics": {
            "normal": {
                "precision": float(precision_score(y_true, y_pred_adjusted, labels=["normal"], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred_adjusted, labels=["normal"], average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred_adjusted, labels=["normal"], average="macro", zero_division=0)),
            },
            "medium": {
                "precision": float(precision_score(y_true, y_pred_adjusted, labels=["medium"], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred_adjusted, labels=["medium"], average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred_adjusted, labels=["medium"], average="macro", zero_division=0)),
            },
            "emergency": {
                "precision": float(precision_score(y_true, y_pred_adjusted, labels=["emergency"], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred_adjusted, labels=["emergency"], average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred_adjusted, labels=["emergency"], average="macro", zero_division=0)),
            },
        },
        "confusion_matrix": cm.tolist(),
    }
    
    (TEST_RESULTS_DIR / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))
    print(f"      ✓ Metrics JSON → metrics_summary.json")
    
    # Save sample predictions (first 10)
    sample_results = {
        "summary": f"Showing first 10 predictions from {len(test_df)} test samples",
        "samples": []
    }
    
    for idx in range(min(10, len(results_df))):
        row = results_df.iloc[idx]
        sample_results["samples"].append({
            "index": idx + 1,
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "correct": bool(row["correct"]),
            "speed_kmh": float(row["speed_kmh"]),
            "auth_failures": int(row["auth_failures"]),
            "anomaly_score": float(row["anomaly_score"]),
            "latency_ms": float(row["latency_ms"]),
        })
    
    (TEST_RESULTS_DIR / "sample_predictions_first10.json").write_text(json.dumps(sample_results, indent=2))
    print(f"      ✓ Sample predictions → sample_predictions_first10.json")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"\n✓ All tests completed successfully!")
    print(f"\nResults saved to: {TEST_RESULTS_DIR}/")
    print(f"\nFiles generated:")
    print(f"  1. sample_test_dataset.csv          - Full test dataset with labels")
    print(f"  2. predictions_with_results.csv     - Predictions + correctness")
    print(f"  3. classification_report.txt        - Detailed metrics per class")
    print(f"  4. confusion_matrix.png             - Visual confusion matrix")
    print(f"  5. metrics_summary.json             - All metrics in JSON format")
    print(f"  6. sample_predictions_first10.json  - First 10 sample predictions")
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  F1-Score:  {f1_macro:.4f}")
    print(f"  Precision: {prec_macro:.4f}")
    print(f"  Recall:    {rec_macro:.4f}")
    print(f"\n✓ Ready to present to your supervisor!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_model()
