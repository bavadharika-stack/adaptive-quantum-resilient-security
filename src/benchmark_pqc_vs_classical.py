"""
Quantum-Resilience Benchmark: PQC (ML-KEM/ML-DSA) vs Classical (ECDH/RSA/ECDSA)
---------------------------------------------------------------------------------
This script does two things that together justify the "quantum resilient"
claim in the paper:

  A) ALGORITHMIC EVIDENCE (the real proof):
     Prints the hardness assumption + best known classical AND quantum
     attack complexity for every scheme used. RSA/ECDH/ECDSA rely on
     integer factorisation / discrete log, both broken in polynomial
     time by Shor's algorithm on a cryptographically relevant quantum
     computer. ML-KEM/ML-DSA rely on Module-LWE / Module-SIS lattice
     problems, for which no efficient quantum algorithm is known -
     the best known quantum attack (quantum lattice sieving) only gives
     a *sub-exponential* speed-up, not a break, which is why NIST
     standardised them (FIPS 203 / FIPS 204, 2024) as quantum-resistant.

  B) EMPIRICAL EVIDENCE (practical cost of switching to PQC):
     Benchmarks wall-clock key generation / encapsulation / signing /
     verification time and key & ciphertext/signature sizes, so the
     paper can report the *overhead* the adaptive framework must
     budget for on emergency-class messages.

Output: reports/quantum_resilience_report.json + .png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pqc_engine import (
    mlkem_keygen, mlkem_encapsulate, mlkem_decapsulate,
    mldsa_keygen, mldsa_sign, mldsa_verify,
    rsa_keygen, rsa_sign, rsa_verify,
    ecdh_keygen, ecdh_exchange,
    ecdsa_keygen, ecdsa_sign, ecdsa_verify,
    timeit_ms,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

ALGORITHM_SECURITY = {
    "RSA-2048": {
        "type": "classical asymmetric encryption/signature",
        "hardness_assumption": "Integer Factorisation Problem (IFP)",
        "classical_attack_bit_security": 112,
        "quantum_attack": "Shor's algorithm - polynomial time (BROKEN by CRQC)",
        "quantum_resistant": False,
    },
    "ECDH-P256 / ECDSA-P256": {
        "type": "classical key exchange / signature",
        "hardness_assumption": "Elliptic Curve Discrete Logarithm Problem (ECDLP)",
        "classical_attack_bit_security": 128,
        "quantum_attack": "Shor's algorithm - polynomial time (BROKEN by CRQC)",
        "quantum_resistant": False,
    },
    "ML-KEM-768": {
        "type": "post-quantum key encapsulation (FIPS 203, 2024)",
        "hardness_assumption": "Module Learning-With-Errors (Module-LWE)",
        "classical_attack_bit_security": 192,
        "quantum_attack": "Best known: quantum lattice sieving - only sub-exponential speed-up, no polynomial break known",
        "quantum_resistant": True,
    },
    "ML-DSA-65": {
        "type": "post-quantum digital signature (FIPS 204, 2024)",
        "hardness_assumption": "Module-LWE / Module-SIS (Fiat-Shamir with Aborts, Dilithium construction)",
        "classical_attack_bit_security": 192,
        "quantum_attack": "Best known: quantum lattice sieving - only sub-exponential speed-up, no polynomial break known",
        "quantum_resistant": True,
    },
    "ASCON-128a": {
        "type": "lightweight symmetric AEAD (NIST LWC standard, 2023)",
        "hardness_assumption": "Symmetric permutation - brute force key search only",
        "classical_attack_bit_security": 128,
        "quantum_attack": "Grover's algorithm halves effective security to ~64 bits; mitigated in the framework by ephemeral, per-session ML-KEM-derived keys and short key lifetimes",
        "quantum_resistant": "Conditionally (needs >=256-bit key or key rotation for long-term quantum resistance; 128-bit key is adequate for short-lived V2X session keys)",
    },
}


def run_benchmarks(repeats=15):
    results = {}

    # ML-KEM-768
    t_keygen, (ek, dk) = timeit_ms(mlkem_keygen, repeats=repeats)
    t_encaps, (ss, ct) = timeit_ms(mlkem_encapsulate, ek, repeats=repeats)
    t_decaps, _ = timeit_ms(mlkem_decapsulate, dk, ct, repeats=repeats)
    results["ML-KEM-768"] = {
        "keygen_ms": t_keygen, "encapsulate_ms": t_encaps, "decapsulate_ms": t_decaps,
        "public_key_bytes": len(ek), "ciphertext_bytes": len(ct),
    }

    # ML-DSA-65
    t_keygen, (pk, sk) = timeit_ms(mldsa_keygen, repeats=repeats)
    msg = b"emergency collision-warning BSM payload"
    t_sign, sig = timeit_ms(mldsa_sign, sk, msg, repeats=repeats)
    t_verify, _ = timeit_ms(mldsa_verify, pk, msg, sig, repeats=repeats)
    results["ML-DSA-65"] = {
        "keygen_ms": t_keygen, "sign_ms": t_sign, "verify_ms": t_verify,
        "public_key_bytes": len(pk), "signature_bytes": len(sig),
    }

    # RSA-2048 (classical signature baseline)
    t_keygen, rk = timeit_ms(rsa_keygen, repeats=max(3, repeats // 5))  # RSA keygen is slow
    t_sign, rsig = timeit_ms(rsa_sign, rk, msg, repeats=repeats)
    t_verify, _ = timeit_ms(rsa_verify, rk.public_key(), msg, rsig, repeats=repeats)
    pub_bytes = rk.public_key().public_numbers().n.bit_length() // 8
    results["RSA-2048"] = {
        "keygen_ms": t_keygen, "sign_ms": t_sign, "verify_ms": t_verify,
        "public_key_bytes": pub_bytes, "signature_bytes": len(rsig),
    }

    # ECDH / ECDSA P-256 (classical baseline)
    t_keygen, ek1 = timeit_ms(ecdh_keygen, repeats=repeats)
    ek2 = ecdh_keygen()
    t_exchange, _ = timeit_ms(ecdh_exchange, ek1, ek2.public_key(), repeats=repeats)
    results["ECDH-P256"] = {"keygen_ms": t_keygen, "exchange_ms": t_exchange}

    t_keygen, sk_ec = timeit_ms(ecdsa_keygen, repeats=repeats)
    t_sign, esig = timeit_ms(ecdsa_sign, sk_ec, msg, repeats=repeats)
    t_verify, _ = timeit_ms(ecdsa_verify, sk_ec.public_key(), msg, esig, repeats=repeats)
    results["ECDSA-P256"] = {
        "keygen_ms": t_keygen, "sign_ms": t_sign, "verify_ms": t_verify,
        "signature_bytes": len(esig),
    }

    return results


def make_plot(bench):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sig_algs = ["ML-DSA-65", "RSA-2048", "ECDSA-P256"]
    sign_times = [bench[a].get("sign_ms", 0) for a in sig_algs]
    verify_times = [bench[a].get("verify_ms", 0) for a in sig_algs]
    x = range(len(sig_algs))
    axes[0].bar([i - 0.2 for i in x], sign_times, width=0.4, label="Sign", color="#c53030")
    axes[0].bar([i + 0.2 for i in x], verify_times, width=0.4, label="Verify", color="#2b6cb0")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(sig_algs, rotation=15)
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Signature Scheme Latency")
    axes[0].legend()

    kem_algs = ["ML-KEM-768", "ECDH-P256"]
    keygen_times = [bench[a]["keygen_ms"] for a in kem_algs]
    op_times = [bench["ML-KEM-768"]["encapsulate_ms"], bench["ECDH-P256"]["exchange_ms"]]
    x2 = range(len(kem_algs))
    axes[1].bar([i - 0.2 for i in x2], keygen_times, width=0.4, label="Keygen", color="#2f855a")
    axes[1].bar([i + 0.2 for i in x2], op_times, width=0.4, label="Encaps/Exchange", color="#d69e2e")
    axes[1].set_xticks(list(x2)); axes[1].set_xticklabels(kem_algs)
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_title("Key-Establishment Latency")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(REPORTS / "pqc_vs_classical_benchmark.png", dpi=150)
    plt.close(fig)


def main():
    bench = run_benchmarks()
    out = {"security_analysis": ALGORITHM_SECURITY, "benchmark_ms_and_bytes": bench}
    (REPORTS / "quantum_resilience_report.json").write_text(json.dumps(out, indent=2))
    make_plot(bench)
    print(json.dumps(out, indent=2))
    print(f"\nSaved -> {REPORTS/'quantum_resilience_report.json'} and pqc_vs_classical_benchmark.png")


if __name__ == "__main__":
    main()
