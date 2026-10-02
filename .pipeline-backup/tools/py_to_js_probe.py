#!/usr/bin/env python3
"""Capture the FULL JS error from the Python→JS decrypt direction."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NODE_DECRYPT = r'''
const p = require("/tmp/dpx/package/dist/index.cjs");
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
try {
  const pt = p.decrypt(Buffer.from(input.priv, "hex"), "encrypted:" + input.body);
  console.log("DECRYPTED:" + Buffer.from(pt).toString());
} catch (e) {
  console.log("JSERROR:" + (e.code || "?") + "::" + e.message);
}
'''


def main() -> int:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

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

    priv = ec.generate_private_key(ec.SECP256K1())
    pv = priv.private_numbers()
    priv_hex = format(pv.private_value, "064x")

    eph = ec.generate_private_key(ec.SECP256K1())
    ev = eph.private_numbers()
    eph_pub = b"\x04" + ev.public_numbers.x.to_bytes(32, "big") + ev.public_numbers.y.to_bytes(32, "big")
    sx, sy = mul(pv.private_value % N, (ev.public_numbers.x, ev.public_numbers.y))
    full_point = b"\x04" + sx.to_bytes(32, "big") + sy.to_bytes(32, "big")
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"", info=None)
    aes_key = hkdf.derive(eph_pub + full_point)
    nonce = os.urandom(16)
    sealed = AESGCM(aes_key).encrypt(nonce, b"python-to-js-456", None)
    blob = base64.b64encode(eph_pub + nonce + sealed).decode()
    payload = json.dumps({"priv": priv_hex, "body": blob})

    r = subprocess.run(["node", "-e", NODE_DECRYPT], input=payload, capture_output=True, text=True)
    out = r.stdout.strip()
    print(out[:400])
    return 0 if "DECRYPTED:python-to-js-456" in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
