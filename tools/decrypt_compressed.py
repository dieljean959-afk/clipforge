#!/usr/bin/env python3
"""Follow-up cross-check: compressed shared point as HKDF input.

The bundled @dotenvx/primitives calls `Point.multiply(r).toBytes()` with NO
argument — noble curves default `toBytes()` to COMPRESSED (33 bytes). This
script verifies that `eph(65) + compressed(33)` is the real HKDF input.

Prints pass/fail only. Run: .venv/bin/python tools/decrypt_compressed.py
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

sys_path = str(Path(__file__).resolve().parent.parent)

NODE = r'''
const p = require("/tmp/dpx/package/dist/index.cjs");
const k = p.keypair();
const ct = p.encrypt(Buffer.from(k.publicKey, "hex"), Buffer.from("roundtrip-test-123", "utf8"));
console.log(JSON.stringify({priv: k.privateKey, body: ct.startsWith("encrypted:") ? ct.slice(10) : ct}));
'''


def main() -> int:
    import sys

    sys.path.insert(0, sys_path)

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    out = subprocess.run(["node", "-e", NODE], capture_output=True, text=True, check=True).stdout
    keys = json.loads(out)
    raw = base64.b64decode(keys["body"] + "=" * (-len(keys["body"]) % 4))
    eph_pub, rest = raw[:65], raw[65:]
    nonce, ct_full = rest[:16], rest[16:]
    tag, ct = ct_full[-16:], ct_full[:-16]

    priv = ec.derive_private_key(int(keys["priv"], 16), ec.SECP256K1())
    eph = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), eph_pub)
    shared_x = priv.exchange(ec.ECDH(), eph)

    # pure-python mult to recover shared Y (parity byte for compression)
    P = 2**256 - 2**32 - 977
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

    def add(p1, p2):
        if p1 is None:
            return p2
        if p1[0] == p2[0] and (p1[1] + p2[1]) % P == 0:
            return (0, 0)
        if p1 == p2:
            lam = (3 * p1[0] * p1[0]) * pow(2 * p1[1], -1, P) % P
        else:
            lam = (p2[1] - p1[1]) * pow(p2[0] - p1[0], -1, P) % P
        x3 = (lam * lam - p1[0] - p2[0]) % P
        return (x3, (lam * (p1[0] - x3) - p1[1]) % P)

    def mul(s, pt):
        res = None
        addend = pt
        while s:
            if s & 1:
                res = add(res, addend)
            addend = add(addend, addend)
            s >>= 1
        return res or (0, 0)

    eph_x = int.from_bytes(eph_pub[1:33], "big")
    eph_y = int.from_bytes(eph_pub[33:65], "big")
    sx, sy = mul(int(keys["priv"], 16) % N, (eph_x, eph_y))
    assert sx.to_bytes(32, "big") == shared_x
    prefix = b"\x03" if (sy & 1) else b"\x02"
    compressed = prefix + sx.to_bytes(32, "big")

    variants = {
        "eph(65) + compressed(33)": eph_pub + compressed,
        "compressed(33) alone": compressed,
        "eph(65) + 0x04XY(65)": eph_pub + b"\x04" + sx.to_bytes(32, "big") + sy.to_bytes(32, "big"),
    }
    winners = []
    for label, hkdf_input in variants.items():
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=None)
        try:
            pt = AESGCM(hkdf.derive(hkdf_input)).decrypt(nonce, ct + tag, None)
            print(f"{label}: SUCCESS -> {pt.decode()!r}")
            winners.append(label)
        except Exception as exc:
            print(f"{label}: fail ({type(exc).__name__})")
    print()
    if winners:
        print("CONCLUSION: HKDF input =", winners[0])
        return 0
    print("NO VARIANT MATCHED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
