#!/usr/bin/env python3
"""Offline cross-check: which derivation matches @dotenvx/primitives?

The official JS lib encrypts a known plaintext with a fresh keypair; this
script decrypts it with several candidate HKDF inputs / shared-point
encodings and reports which succeeds. Prints pass/fail only, no key material.

Run: .venv/bin/python tools/decrypt_variants.py   (requires node + /tmp/dpx)
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NODE = r'''
const p = require("/tmp/dpx/package/dist/index.cjs");
const k = p.keypair();
const ct = p.encrypt(Buffer.from(k.publicKey, "hex"), Buffer.from("roundtrip-test-123", "utf8"));
console.log(JSON.stringify({priv: k.privateKey, pub: k.publicKey, body: ct.startsWith("encrypted:") ? ct.slice(10) : ct}));
'''


def main() -> int:
    out = subprocess.run(["node", "-e", NODE], capture_output=True, text=True, check=True).stdout
    keys = json.loads(out)

    raw = base64.b64decode(keys["body"] + "=" * (-len(keys["body"]) % 4))
    eph_pub = raw[:65]                      # 0x04 || X || Y
    nonce = raw[65:81]
    ct_full = raw[81:]
    tag, ct = ct_full[-16:], ct_full[:-16]

    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    # compute shared point via the cryptography lib (scalar * eph_pub)
    priv = ec.derive_private_key(int(keys["priv"], 16), ec.SECP256K1())
    eph = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), eph_pub)
    shared_x = priv.exchange(ec.ECDH(), eph)          # 32-byte X of shared point
    # recover shared Y via scalar mult on our own (same math as app.config)
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
    assert sx.to_bytes(32, "big") == shared_x, "own scalar mult disagrees with cryptography lib"

    eph_pub_compressed = eph_pub[0:1] + eph_pub[1:33]

    variants = {
        "eph(65) + XY(64)  [current app]": eph_pub + sx.to_bytes(32, "big") + sy.to_bytes(32, "big"),
        "eph(65) + 0x04XY(65)": eph_pub + b"\x04" + sx.to_bytes(32, "big") + sy.to_bytes(32, "big"),
        "ephCompressed(33) + XY(64)": eph_pub_compressed + sx.to_bytes(32, "big") + sy.to_bytes(32, "big"),
        "ephXY(64) + XY(64)": eph_pub[1:] + sx.to_bytes(32, "big") + sy.to_bytes(32, "big"),
        "eph(65) + X(32) only": eph_pub + shared_x,
        "X(32) only": shared_x,
    }

    winners = []
    for label, hkdf_input in variants.items():
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=None)
        aes_key = hkdf.derive(hkdf_input)
        try:
            pt = AESGCM(aes_key).decrypt(nonce, ct + tag, None)
            print(f"{label}: SUCCESS -> {pt.decode()!r}")
            winners.append(label)
        except Exception as exc:
            print(f"{label}: fail ({type(exc).__name__})")

    print()
    if winners:
        print("CONCLUSION: use", winners[0])
        return 0
    print("NO VARIANT MATCHED — derivation differs in another way.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
