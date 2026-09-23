#!/usr/bin/env python3
"""DEFINITIVE cross-check of the dotenvx decrypt fix in app/config.py.

Direction 1 (JS encrypt → Python decrypt): the official @dotenvx/primitives
encrypts a fresh keypair + known plaintext; app.config._dotenvx_decrypt must
recover it.

Direction 2 (Python encrypt → JS decrypt): we build the ciphertext with the
same construction; the official lib must decrypt it.

Run: .venv/bin/python tools/decrypt_confirm.py   (requires node + /tmp/dpx)
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import _dotenvx_decrypt  # noqa: E402

NODE_ROUNDTRIP = r'''
const p = require("/tmp/dpx/package/dist/index.cjs");
const mode = process.argv[1];  // node -e puts the first extra arg at argv[1]
if (mode === "encrypt") {
  // encrypt; emit keys + ciphertext body for python to decrypt
  const k = p.keypair();
  const ct = p.encrypt(Buffer.from(k.publicKey, "hex"), Buffer.from("roundtrip-test-123", "utf8"));
  console.log(JSON.stringify({priv: k.privateKey, body: ct.startsWith("encrypted:") ? ct.slice(10) : ct}));
} else {
  // decrypt: python produced {pub, body}; verify
  const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
  const pt = p.decrypt(Buffer.from(input.priv, "hex"), "encrypted:" + input.body);
  console.log(Buffer.from(pt).toString());
}
'''


def main() -> int:
    # ---- direction 1: JS encrypt -> python decrypt
    out = subprocess.run(
        ["node", "-e", NODE_ROUNDTRIP, "encrypt"], capture_output=True, text=True, check=True
    ).stdout
    keys = json.loads(out)
    got = _dotenvx_decrypt(keys["priv"], keys["body"])
    print(f"JS encrypt -> Python decrypt: {'PASS' if got == 'roundtrip-test-123' else 'FAIL: ' + got!r}")
    ok1 = got == "roundtrip-test-123"

    # NOTE the dotenvx blob layout: eph(65) || nonce(16) || TAG(16) || ct
    # (the GCM tag comes BEFORE the ciphertext — see symEncrypt in the lib).

    # ---- direction 2: python encrypt -> JS decrypt
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    import os

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

    def py_encrypt(priv_hex: str, plaintext: bytes) -> str:
        eph = ec.generate_private_key(ec.SECP256K1())
        eph_pub = b"\x04" + eph.private_numbers().public_numbers.x.to_bytes(32, "big") + eph.private_numbers().public_numbers.y.to_bytes(32, "big")
        eph_x = eph.private_numbers().public_numbers.x
        eph_y = eph.private_numbers().public_numbers.y
        sx, sy = mul(int(priv_hex, 16) % N, (eph_x, eph_y))
        full_point = b"\x04" + sx.to_bytes(32, "big") + sy.to_bytes(32, "big")
        hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"", info=None)
        aes_key = hkdf.derive(eph_pub + full_point)
        nonce = os.urandom(16)
        sealed = AESGCM(aes_key).encrypt(nonce, plaintext, None)  # ct || tag
        ct, tag = sealed[:-16], sealed[-16:]
        blob = eph_pub + nonce + tag + ct  # dotenvx layout: nonce || TAG || ct
        return base64.b64encode(blob).decode()

    from cryptography.hazmat.primitives import hashes  # noqa: F401

    priv = ec.generate_private_key(ec.SECP256K1())
    priv_hex = format(priv.private_numbers().private_value, "064x")
    pub_hex = "04" + format(priv.private_numbers().public_numbers.x, "064x") + format(priv.private_numbers().public_numbers.y, "064x")
    body = py_encrypt(priv_hex, b"python-to-js-456")
    payload = json.dumps({"priv": priv_hex, "body": body})
    got2 = subprocess.run(
        ["node", "-e", NODE_ROUNDTRIP, "decrypt"], input=payload, capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"Python encrypt -> JS decrypt: {'PASS' if got2 == 'python-to-js-456' else 'FAIL: ' + got2!r}")
    ok2 = got2 == "python-to-js-456"

    print()
    if ok1 and ok2:
        print("CONCLUSION: app.config._dotenvx_decrypt is now byte-compatible with")
        print("@dotenvx/primitives in BOTH directions.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
