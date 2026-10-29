"""
Adaptive Quantum-Resilient Lightweight Security Framework for IoV
=====================================================================
Single entry point that runs the full prototype pipeline end-to-end:

  Step 1: Generate the IoV network telemetry dataset
  Step 2: Train + evaluate the ML risk classifier (normal/medium/emergency)
  Step 3: Run the adaptive security demo (ASCON / ML-KEM / ML-DSA selection)
  Step 4: Benchmark PQC vs classical crypto and write the quantum
          resilience evidence report

Run with:  python main.py
All outputs land in ./reports and ./data
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run(script):
    print(f"\n{'='*70}\nRUNNING: {script}\n{'='*70}")
    result = subprocess.run([sys.executable, str(SRC / script)], cwd=str(SRC))
    if result.returncode != 0:
        print(f"FAILED: {script}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    run("generate_dataset.py")
    run("train_model.py")
    run("adaptive_security.py")
    run("benchmark_pqc_vs_classical.py")
    print(f"\n\nAll done. See the '{ROOT/'reports'}' folder for:")
    print("  - model.pkl, metrics_summary.json, classification_report.txt")
    print("  - confusion_matrix.png, feature_importance.png, class_distribution.png")
    print("  - adaptive_demo_output.json")
    print("  - quantum_resilience_report.json, pqc_vs_classical_benchmark.png")
