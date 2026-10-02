#!/usr/bin/env python3
"""End-to-end dotenvx decrypt test — verifies the app's decryptor round-trips.

Encrypts known plaintext with the OFFICIAL @dotenvx/primitives lib (node),
decrypts it with the app's pure-Python _dotenvx_decrypt, and asserts equality.
Prints only pass/fail + lengths — no key material, no plaintext.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import _dotenvx_decrypt  # noqa: E402

NODE_ENCRYPT = r"""
const p = require("/tmp/dpx/package/dist/index.cjs");
const { publicKey, plaintext } = JSON.parse(require("fs").readFileSync(0, "utf8"));
// p.encrypt returns 'encrypted:<base64>' — strip the prefix, keep the b64 body
const ct = p.encrypt(Buffer.from(publicKey, "hex"), Buffer.from(plaintext, "utf8"));
console.log(ct.startsWith("encrypted:") ? ct.slice("encrypted:".length) : ct);
"""


def main() -> int:
    # 1. generate a fresh keypair with the official lib
    kp = subprocess.run(
        ["node", "-e",
         'const p=require("/tmp/dpx/package/dist/index.cjs");'
         'const k=p.keypair();console.log(JSON.stringify({publicKey:k.publicKey,privateKey:k.privateKey}))'],
        capture_output=True, text=True, check=True,
    )
    keys = json.loads(kp.stdout)

    # 2. encrypt a known plaintext with the official lib
    plaintext = "AIzaTest-Key_1234567890-abcdef"
    ct = subprocess.run(
        ["node", "-e", NODE_ENCRYPT],
        input=json.dumps({"publicKey": keys["publicKey"], "plaintext": plaintext}),
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    print(f"official encrypt → ciphertext {len(ct)} chars (b64)")

    # 3. decrypt with the app's pure-Python implementation
    got = _dotenvx_decrypt(keys["privateKey"], ct)
    print(f"python decrypt → {len(got)} chars, match={got == plaintext}")
    if got != plaintext:
        print("MISMATCH — implementation diverges from @dotenvx/primitives")
        return 1

    # 4. empty-plaintext edge case (what the platform currently stores)
    ct_empty = subprocess.run(
        ["node", "-e", NODE_ENCRYPT],
        input=json.dumps({"publicKey": keys["publicKey"], "plaintext": ""}),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    got_empty = _dotenvx_decrypt(keys["privateKey"], ct_empty)
    print(f"empty-plaintext edge case: decrypts to {len(got_empty)} chars (expected 0)")
    return 0 if got_empty == "" else 1


if __name__ == "__main__":
    raise SystemExit(main())
