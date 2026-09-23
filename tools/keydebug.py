#!/usr/bin/env python3
"""Debug dotenvx decryption failure — prints error TYPE and structural facts
(blob segment lengths, key name, curve expectations), never secret bytes."""
from __future__ import annotations

import base64
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    for name in (".env", ".env.local", ".env.keys"):
        p = ROOT / name
        if p.exists():
            load_dotenv(p, override=False)

    enc = os.environ.get("GEMINI_API_KEY", "")
    priv = os.environ.get("DOTENV_PRIVATE_KEY_LOCAL", "").strip()

    print(f"encrypted value present: {bool(enc)}, len {len(enc)}")
    print(f"private key present: {bool(priv)}, len {len(priv)}")
    if not enc or not priv:
        return 3

    body = enc[len("encrypted:"):] if enc.startswith("encrypted:") else enc
    try:
        raw = base64.b64decode(body + "=" * (-len(body) % 4))
        print(f"base64 decoded: {len(raw)} bytes")
    except Exception as exc:
        print(f"base64 decode failed: {type(exc).__name__}: {exc}")
        return 1

    # try different ephemeral-key splits
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    for eph_len in (65, 33):
        for nonce_len in (16, 12):
            eph = raw[:eph_len]
            rest = raw[eph_len:]
            nonce, rest2 = rest[:nonce_len], rest[nonce_len:]
            tag, ct = rest2[:16], rest2[16:]
            try:
                privk = ec.derive_private_key(int(priv, 16), ec.SECP256K1())
                pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), eph)
                shared = privk.exchange(ec.ECDH(), pub)
                hk = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=None)
                aes = hk.derive(eph + shared)
                AESGCM(aes).decrypt(nonce, ct + tag, None)
                print(f"SUCCESS with eph={eph_len} nonce={nonce_len}")
                return 0
            except ValueError as exc:
                print(f"eph={eph_len} nonce={nonce_len}: ValueError: {str(exc)[:90]}")
            except Exception as exc:
                print(f"eph={eph_len} nonce={nonce_len}: {type(exc).__name__}: {str(exc)[:90]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
