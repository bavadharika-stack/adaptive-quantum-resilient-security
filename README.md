# Adaptive Quantum-Resilient Security Framework for IoV

**An adaptive security framework for the Internet of Vehicles (IoV)** combining lightweight cryptography, post-quantum cryptography, and machine learning-based risk classification.

## Overview

This prototype classifies network messages into three risk levels — normal, medium, and emergency — and selects a corresponding cryptographic protection policy.

The framework integrates:
- **ASCON** — lightweight authenticated encryption.
- **ML-KEM (FIPS 203)** — post-quantum key encapsulation.
- **ML-DSA (FIPS 204)** — post-quantum digital signatures.
- **Machine Learning** — risk classification for adaptive security decisions.
- **RSA and ECC** — classical cryptographic algorithms for comparison.

## Project Structure

```text
IoV_Quantum_Security_Project/
├── main.py
├── requirements.txt
├── src/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── pqc_engine.py
│   ├── adaptive_security.py
│   └── benchmark_pqc_vs_classical.py
├── data/
└── reports/
```

*The actual directory and file names may vary slightly depending on the project version.*

## Getting Started

Requirements: Python 3.10 or later.

1. Clone or download this repository.
2. Open the project directory in VS Code.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Run the main pipeline:

   ```powershell
   python main.py
   ```

Run individual scripts from the `src/` directory when needed.

## Evaluation

The prototype is designed to evaluate:
- Risk-classification performance.
- Adaptive cryptographic policy selection.
- Cryptographic operation correctness.
- Performance comparisons between post-quantum and classical algorithms.

Report actual experimental results only after reproducing and verifying them.

## Limitations

The framework's quantum-resistance claims are based on the security assumptions of the selected post-quantum algorithms, not on a live quantum-attack demonstration. Synthetic datasets, pure-Python implementations, and simulated deployment conditions may limit how directly results transfer to real-world automotive systems.

## Future Work

- Evaluate against public IoV intrusion-detection datasets.
- Benchmark optimized cryptographic implementations.
- Test under realistic vehicle-network conditions.
- Investigate deployment constraints for automotive hardware.
