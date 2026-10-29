"""
Adaptive Quantum-Resilient Security Framework (Prototype)
-----------------------------------------------------------
End-to-end pipeline:

  1. A V2X message arrives with its telemetry features.
  2. The trained ML risk classifier (reports/model.pkl) predicts
     risk_level in {normal, medium, emergency}.
  3. Based on the predicted risk, the SECURITY POLICY selects which
     cryptographic primitives protect the message:

        normal    -> ASCON-128a only
                     (max speed, message is low-value beaconing data)

        medium    -> ML-KEM-768 (fresh session key) + ASCON-128a
                     (adds post-quantum key freshness for
                      moderately sensitive / congested-network traffic)

        emergency -> ML-KEM-768 + ML-DSA-65 + ASCON-128a
                     (adds a post-quantum signature so the receiving
                      vehicle/RSU can authenticate a safety-critical
                      message and defend against spoofing/replay,
                      which is exactly the attack class observed in the
                      'emergency' portion of the dataset)

  4. Each stage is timed so the report can quantify the
     security-vs-latency trade-off the dissertation/paper needs.

This file can be run directly for a demo, and also exposes
`secure_message()` for reuse (e.g. by a Flask/FastAPI wrapper).
"""
import json
import secrets
import time
from pathlib import Path

import joblib  # type: ignore
import pandas as pd  # type: ignore

from pqc_engine import (  # type: ignore
    ascon_encrypt, ascon_decrypt,
    mlkem_keygen, mlkem_encapsulate, mlkem_decapsulate,
    mldsa_keygen, mldsa_sign, mldsa_verify,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "reports" / "model.pkl"
FEATURES = [
    "speed_kmh", "message_priority", "packet_size_bytes", "latency_ms",
    "packet_loss_pct", "network_load_pct", "distance_m",
    "message_frequency_hz", "auth_failures", "anomaly_score",
]

# Pre-generate a long-lived ML-DSA identity keypair per vehicle/RSU
# (in a real deployment this is provisioned once, not per-message)
_MLDSA_PK, _MLDSA_SK = mldsa_keygen()


def classify_risk(model, feature_row: dict) -> str:
    X = pd.DataFrame([feature_row])[FEATURES]
    
    # Use probability-based classification with adjusted threshold for emergency
    # This improves recall for the safety-critical emergency class
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    
    # Get emergency class probability
    emergency_idx = list(classes).index("emergency")
    emergency_prob = proba[emergency_idx]
    
    # If emergency probability is reasonably high (>25%), predict emergency
    # This lowers the threshold from default 33.3% (1/3 classes) to 25%
    # to catch more safety-critical cases
    if emergency_prob > 0.25:
        return "emergency"
    
    # Otherwise use standard prediction
    return model.predict(X)[0]


def secure_message(model, feature_row: dict, message: bytes):
    """
    Applies the adaptive policy to a single V2X message.
    Returns a dict with the risk level, chosen policy, ciphertext,
    and a per-stage latency breakdown (all in milliseconds).
    """
    timings = {}

    t0 = time.perf_counter()
    risk = classify_risk(model, feature_row)
    timings["ml_classification_ms"] = (time.perf_counter() - t0) * 1000

    session_key = secrets.token_bytes(16)   # will possibly be replaced by KEM output
    nonce = secrets.token_bytes(16)
    kem_ciphertext = None
    signature = None

    if risk == "normal":
        policy = "ASCON-128a"

    elif risk == "medium":
        policy = "ML-KEM-768 + ASCON-128a"
        t0 = time.perf_counter()
        ek, dk = mlkem_keygen()
        shared_secret, kem_ciphertext = mlkem_encapsulate(ek)
        session_key = shared_secret[:16]
        timings["mlkem_keyestablish_ms"] = (time.perf_counter() - t0) * 1000

    else:  # emergency
        policy = "ML-KEM-768 + ML-DSA-65 + ASCON-128a"
        t0 = time.perf_counter()
        ek, dk = mlkem_keygen()
        shared_secret, kem_ciphertext = mlkem_encapsulate(ek)
        session_key = shared_secret[:16]
        timings["mlkem_keyestablish_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        signature = mldsa_sign(_MLDSA_SK, message)
        timings["mldsa_sign_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ciphertext = ascon_encrypt(session_key, nonce, message)
    timings["ascon_encrypt_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(timings.values())

    return {
        "risk_level": risk,
        "policy": policy,
        "ciphertext": ciphertext.hex(),
        "nonce": nonce.hex(),
        "kem_ciphertext_len_bytes": len(kem_ciphertext) if kem_ciphertext else 0,
        "signature_len_bytes": len(signature) if signature else 0,
        "signature_valid": mldsa_verify(_MLDSA_PK, message, signature) if signature else None,
        "timings_ms": timings,
    }


def demo():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(ROOT / "data" / "iov_security_dataset.csv")

    # Pick a representative example per class using the model's own
    # confidence, so the demo doesn't accidentally showcase one of the
    # ~3% intentionally noisy/mislabeled rows in the dataset.
    examples = []
    for level in ["normal", "medium", "emergency"]:
        subset = df[df["risk_level"] == level].copy()
        X_subset = subset[FEATURES]
        proba = model.predict_proba(X_subset)
        class_idx = list(model.classes_).index(level)
        subset["_confidence"] = proba[:, class_idx]
        best_row = subset.sort_values("_confidence", ascending=False).iloc[0]
        examples.append(best_row.drop("_confidence").to_dict())

    results = []
    for i, row in enumerate(examples):
        print(f"[LOOP] Processing example {i}, true_label={row['risk_level']}", flush=True)
        msg = f"BSM speed={row['speed_kmh']} dist={row['distance_m']}".encode()
        
        # DEBUG: Check probabilities before classification
        if i == 2:  # emergency sample
            X_debug = pd.DataFrame([row])[FEATURES]
            proba_debug = model.predict_proba(X_debug)[0]
            print(f"\n[DEBUG] Emergency sample: true_label={row['risk_level']}", flush=True)
            for j, cls in enumerate(model.classes_):
                print(f"[DEBUG]   {cls}: {proba_debug[j]:.4f}", flush=True)
        
        result = secure_message(model, row, msg)
        results.append(result)
        print(f"\n--- Sample record (true label: {row['risk_level']}) ---")
        print(f"Predicted risk : {result['risk_level']}")
        print(f"Policy applied : {result['policy']}")
        print(f"Timings (ms)   : {json.dumps(result['timings_ms'], indent=2)}")
        if result["signature_valid"] is not None:
            print(f"ML-DSA signature valid: {result['signature_valid']}")

    out_path = ROOT / "reports" / "adaptive_demo_output.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved demo output -> {out_path}")


if __name__ == "__main__":
    demo()