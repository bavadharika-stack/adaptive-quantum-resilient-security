"""
IoV (Internet of Vehicles) Network Security Telemetry Dataset Generator
-------------------------------------------------------------------------
Generates a realistic (physics- and protocol-grounded, not purely random)
dataset that mimics V2X / VANET message telemetry for the purpose of
training a risk-classification model that feeds an adaptive, quantum
resilient security framework (ASCON + ML-KEM + ML-DSA).

Design principles used to avoid a "too synthetic" dataset:
  1. Every feature is generated from a real-world plausible range taken
     from V2X/DSRC/C-V2X and IEEE 1609.2 literature (not arbitrary).
  2. Features are CORRELATED the way they are in real vehicular networks,
     e.g. higher speed -> higher Doppler-driven packet loss & latency;
     higher network load -> higher latency & loss; jamming/spoofing
     attacks raise auth_failures & anomaly_score together, not independently.
  3. Three realistic *situation classes* are modelled with their own
     generative sub-process (not just thresholds on one column):
       - NORMAL   : healthy V2V/V2I beaconing traffic
       - MEDIUM   : degraded-but-not-critical conditions (congestion,
                    marginal signal, partial jamming, minor auth issues)
       - EMERGENCY: safety-critical events (collision warning, RSU under
                    attack, GPS spoofing, DoS/flooding, replay attacks)
  4. Label noise (~3%) and sensor noise are injected so the classes are
     NOT linearly / trivially separable -- this forces the ML model to
     learn genuine structure instead of memorising a threshold rule.
  5. Class imbalance mirrors reality: EMERGENCY events are rare.

Output: data/iov_security_dataset.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

N_TOTAL = 12000

# Realistic class prior (rare emergencies, common normal traffic)
CLASS_PROPORTIONS = {"normal": 0.62, "medium": 0.30, "emergency": 0.08}


def clip(a, lo, hi):
    return np.clip(a, lo, hi)


def gen_normal(n):
    """Healthy V2V/V2I beaconing traffic (SAE J2735 BSM-like)."""
    speed_kmh = clip(rng.normal(60, 18, n), 0, 140)
    message_priority = rng.choice([1, 2, 3], size=n, p=[0.70, 0.25, 0.05])
    packet_size_bytes = clip(rng.normal(300, 40, n), 100, 500)
    distance_m = clip(rng.normal(150, 60, n), 5, 500)
    network_load_pct = clip(rng.normal(35, 12, n), 2, 100)
    message_frequency_hz = clip(rng.normal(10, 1.5, n), 1, 20)

    # latency grows with distance, network load and a Doppler term from speed
    latency_ms = (
        5
        + 0.02 * distance_m
        + 0.25 * network_load_pct
        + 0.03 * speed_kmh
        + rng.normal(0, 2.5, n)
    )
    latency_ms = clip(latency_ms, 1, 400)

    packet_loss_pct = clip(
        0.01 * network_load_pct + 0.01 * (speed_kmh / 10) + rng.normal(0, 1.0, n),
        0, 15,
    )
    auth_failures = rng.poisson(0.05, n)
    anomaly_score = clip(
        0.02 * network_load_pct + 0.3 * auth_failures + rng.normal(5, 3, n), 0, 100
    )
    return dict(
        speed_kmh=speed_kmh, message_priority=message_priority,
        packet_size_bytes=packet_size_bytes, latency_ms=latency_ms,
        packet_loss_pct=packet_loss_pct, network_load_pct=network_load_pct,
        distance_m=distance_m, message_frequency_hz=message_frequency_hz,
        auth_failures=auth_failures, anomaly_score=anomaly_score,
    )


def gen_medium(n):
    """Degraded conditions: congestion, marginal RF link, minor auth issues."""
    speed_kmh = clip(rng.normal(75, 25, n), 0, 160)
    message_priority = rng.choice([1, 2, 3], size=n, p=[0.30, 0.45, 0.25])
    packet_size_bytes = clip(rng.normal(340, 60, n), 100, 800)
    distance_m = clip(rng.normal(280, 90, n), 20, 900)
    network_load_pct = clip(rng.normal(65, 15, n), 20, 100)
    message_frequency_hz = clip(rng.normal(8, 3, n), 0.5, 20)

    latency_ms = (
        20
        + 0.05 * distance_m
        + 0.55 * network_load_pct
        + 0.08 * speed_kmh
        + rng.normal(0, 6, n)
    )
    latency_ms = clip(latency_ms, 5, 700)

    packet_loss_pct = clip(
        0.06 * network_load_pct + 0.03 * (speed_kmh / 10) + rng.normal(2, 2.5, n),
        0, 40,
    )
    auth_failures = rng.poisson(0.8, n)
    anomaly_score = clip(
        0.35 * network_load_pct + 3.0 * auth_failures + rng.normal(20, 8, n), 0, 100
    )
    return dict(
        speed_kmh=speed_kmh, message_priority=message_priority,
        packet_size_bytes=packet_size_bytes, latency_ms=latency_ms,
        packet_loss_pct=packet_loss_pct, network_load_pct=network_load_pct,
        distance_m=distance_m, message_frequency_hz=message_frequency_hz,
        auth_failures=auth_failures, anomaly_score=anomaly_score,
    )


def gen_emergency(n):
    """
    Safety-critical / attack-affected traffic: collision-imminent beacons,
    RSU flooding / DoS, GPS-spoofed position jumps, replay & Sybil style
    authentication storms. Mixture of 3 emergency sub-scenarios.
    """
    scenario = rng.choice(["collision", "dos_jam", "spoof_replay"], size=n,
                           p=[0.45, 0.30, 0.25])

    speed_kmh = np.zeros(n)
    packet_size_bytes = np.zeros(n)
    distance_m = np.zeros(n)
    network_load_pct = np.zeros(n)
    message_frequency_hz = np.zeros(n)
    auth_failures = np.zeros(n)
    latency_ms = np.zeros(n)
    packet_loss_pct = np.zeros(n)
    anomaly_score = np.zeros(n)
    message_priority = np.ones(n, dtype=int) * 3  # emergencies are top priority

    is_col = scenario == "collision"
    n_col = is_col.sum()
    speed_kmh[is_col] = clip(rng.normal(100, 25, n_col), 20, 200)
    distance_m[is_col] = clip(rng.normal(15, 8, n_col), 1, 60)  # imminent hazard, close range
    network_load_pct[is_col] = clip(rng.normal(55, 15, n_col), 10, 100)
    message_frequency_hz[is_col] = clip(rng.normal(15, 3, n_col), 5, 20)  # rapid re-beaconing
    packet_size_bytes[is_col] = clip(rng.normal(320, 50, n_col), 100, 600)
    auth_failures[is_col] = clip(rng.poisson(2.5, n_col) + 2, 2, 20)  # minimum 2 auth failures
    latency_ms[is_col] = clip(15 + 0.3 * distance_m[is_col] + rng.normal(5, 4, n_col), 1, 200)
    packet_loss_pct[is_col] = clip(rng.normal(8, 5, n_col), 0, 40)
    anomaly_score[is_col] = clip(rng.normal(70, 12, n_col), 35, 100)  # minimum 35 anomaly

    is_dos = scenario == "dos_jam"
    n_dos = is_dos.sum()
    speed_kmh[is_dos] = clip(rng.normal(55, 25, n_dos), 0, 150)
    distance_m[is_dos] = clip(rng.normal(200, 100, n_dos), 5, 800)
    network_load_pct[is_dos] = clip(rng.normal(95, 6, n_dos), 70, 100)  # flooded channel
    message_frequency_hz[is_dos] = clip(rng.normal(19, 4, n_dos), 5, 40)  # attacker flood
    packet_size_bytes[is_dos] = clip(rng.normal(500, 150, n_dos), 100, 1200)
    auth_failures[is_dos] = rng.poisson(4.5, n_dos)
    latency_ms[is_dos] = clip(rng.normal(350, 120, n_dos), 50, 900)
    packet_loss_pct[is_dos] = clip(rng.normal(45, 18, n_dos), 10, 95)
    anomaly_score[is_dos] = clip(rng.normal(85, 10, n_dos), 40, 100)

    is_sp = scenario == "spoof_replay"
    n_sp = is_sp.sum()
    speed_kmh[is_sp] = clip(rng.normal(70, 40, n_sp), 0, 220)  # implausible jumps
    distance_m[is_sp] = clip(rng.normal(120, 90, n_sp), 1, 900)
    network_load_pct[is_sp] = clip(rng.normal(60, 20, n_sp), 10, 100)
    message_frequency_hz[is_sp] = clip(rng.normal(12, 5, n_sp), 1, 30)
    packet_size_bytes[is_sp] = clip(rng.normal(300, 80, n_sp), 100, 900)
    auth_failures[is_sp] = clip(rng.poisson(6.0, n_sp) + 3, 3, 25)  # minimum 3 auth failures
    latency_ms[is_sp] = clip(rng.normal(120, 60, n_sp), 5, 500)
    packet_loss_pct[is_sp] = clip(rng.normal(20, 12, n_sp), 0, 70)
    anomaly_score[is_sp] = clip(rng.normal(90, 8, n_sp), 60, 100)  # minimum 60 anomaly

    return dict(
        speed_kmh=speed_kmh, message_priority=message_priority,
        packet_size_bytes=packet_size_bytes, latency_ms=latency_ms,
        packet_loss_pct=packet_loss_pct, network_load_pct=network_load_pct,
        distance_m=distance_m, message_frequency_hz=message_frequency_hz,
        auth_failures=auth_failures.astype(int), anomaly_score=anomaly_score,
    )


def build_dataset():
    n_normal = int(N_TOTAL * CLASS_PROPORTIONS["normal"])
    n_medium = int(N_TOTAL * CLASS_PROPORTIONS["medium"])
    n_emerg = N_TOTAL - n_normal - n_medium

    d_n = gen_normal(n_normal)
    d_m = gen_medium(n_medium)
    d_e = gen_emergency(n_emerg)

    labels = (
        ["normal"] * n_normal + ["medium"] * n_medium + ["emergency"] * n_emerg
    )

    df = pd.DataFrame({k: np.concatenate([d_n[k], d_m[k], d_e[k]]) for k in d_n})
    df["risk_level"] = labels

    # shuffle rows
    df = df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

    # ---- realism touches ----
    # 1) round/clean columns to realistic precision & integer semantics
    df["message_priority"] = df["message_priority"].astype(int)
    df["packet_size_bytes"] = df["packet_size_bytes"].round(0).astype(int)
    df["auth_failures"] = df["auth_failures"].round(0).astype(int)
    for c in ["speed_kmh", "latency_ms", "packet_loss_pct", "network_load_pct",
              "distance_m", "message_frequency_hz", "anomaly_score"]:
        df[c] = df[c].round(2)

    # 2) inject small missing-value rate on a non-critical sensor field,
    #    exactly as real IoT telemetry has occasional dropped readings
    miss_idx = rng.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[miss_idx, "message_frequency_hz"] = np.nan
    df["message_frequency_hz"] = df["message_frequency_hz"].fillna(
        df["message_frequency_hz"].median()
    )

    # 3) label noise: flip ~3% of labels to a neighbouring class to avoid
    #    a trivially-separable synthetic dataset
    noise_idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
    order = ["normal", "medium", "emergency"]
    for i in noise_idx:
        cur = df.at[i, "risk_level"]
        choices = [c for c in order if c != cur]
        df.at[i, "risk_level"] = rng.choice(choices)

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    df = build_dataset()
    out_path = out_dir / "iov_security_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows -> {out_path}")
    print(df["risk_level"].value_counts())
    print(df.describe(include="all").T)
