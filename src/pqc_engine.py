"""
PQC / Lightweight Crypto Engine
--------------------------------
Real (not simulated) implementations of the three primitives used by the
Adaptive Quantum-Resilient Lightweight Security Framework:

  - ASCON-128a       : NIST Lightweight Cryptography standard (2023) - AEAD
  - ML-KEM-768        : FIPS 203 (2024) - post-quantum key encapsulation
                        (module-lattice / Kyber family)
  - ML-DSA-65         : FIPS 204 (2024) - post-quantum digital signature
                        (module-lattice / Dilithium family)

Classical baselines (RSA-2048, ECDSA P-256 / ECDH) are also implemented
using `cryptography` so we can benchmark PQC vs classical overhead and
support the quantum-resilience discussion in the report.
"""
import time

import ascon
from kyber_py.ml_kem import ML_KEM_768
from dilithium_py.ml_dsa import ML_DSA_65

from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization


# ---------------------------------------------------------------------
# 1. ASCON (lightweight symmetric AEAD) - used for EVERY message
# ---------------------------------------------------------------------
def ascon_encrypt(key: bytes, nonce: bytes, plaintext: bytes, associated_data: bytes = b""):
    return ascon.encrypt(key, nonce, associated_data, plaintext, variant="Ascon-128a")


def ascon_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, associated_data: bytes = b""):
    return ascon.decrypt(key, nonce, associated_data, ciphertext, variant="Ascon-128a")


# ---------------------------------------------------------------------
# 2. ML-KEM-768 (post-quantum key establishment) - used for MEDIUM/EMERGENCY
# ---------------------------------------------------------------------
def mlkem_keygen():
    return ML_KEM_768.keygen()  # (encapsulation_key, decapsulation_key)


def mlkem_encapsulate(ek):
    return ML_KEM_768.encaps(ek)  # (shared_secret, ciphertext)


def mlkem_decapsulate(dk, ct):
    return ML_KEM_768.decaps(dk, ct)


# ---------------------------------------------------------------------
# 3. ML-DSA-65 (post-quantum digital signature) - used for EMERGENCY only
# ---------------------------------------------------------------------
def mldsa_keygen():
    return ML_DSA_65.keygen()  # (public_key, secret_key)


def mldsa_sign(sk, message: bytes):
    return ML_DSA_65.sign(sk, message)


def mldsa_verify(pk, message: bytes, sig: bytes) -> bool:
    return ML_DSA_65.verify(pk, message, sig)


# ---------------------------------------------------------------------
# Classical baselines, for the quantum-resilience comparison only
# ---------------------------------------------------------------------
def rsa_keygen(bits=2048):
    return rsa.generate_private_key(public_exponent=65537, key_size=bits)


def rsa_sign(private_key, message: bytes):
    return private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )


def rsa_verify(public_key, message: bytes, sig: bytes) -> bool:
    try:
        public_key.verify(
            sig, message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def ecdh_keygen():
    return ec.generate_private_key(ec.SECP256R1())


def ecdh_exchange(priv, peer_pub):
    return priv.exchange(ec.ECDH(), peer_pub)


def ecdsa_keygen():
    return ec.generate_private_key(ec.SECP256R1())


def ecdsa_sign(priv, message: bytes):
    return priv.sign(message, ec.ECDSA(hashes.SHA256()))


def ecdsa_verify(pub, message: bytes, sig: bytes) -> bool:
    try:
        pub.verify(sig, message, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------
def timeit_ms(fn, *args, repeats=20, **kwargs):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), result
