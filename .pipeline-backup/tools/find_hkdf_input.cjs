// Find the exact HKDF input @dotenvx/primitives uses by capturing the real
// AES key during decrypt and testing candidate IKM constructions (including
// the full uncompressed shared point computed with BigInt EC math).
// Prints only MATCH/no verdicts and lengths — no secret material.
const p = require("/tmp/dpx/package/dist/index.cjs");
const crypto = require("crypto");

const orig = crypto.createDecipheriv;
let aesKey = null;
crypto.createDecipheriv = function (alg, key, iv, opts) {
  if (alg === "aes-256-gcm" && !aesKey) aesKey = Buffer.from(key);
  return orig.call(crypto, alg, key, iv, opts);
};

const k = p.keypair();
const msg = Buffer.from("roundtrip-test-123", "utf8");
const ct = p.encrypt(Buffer.from(k.publicKey, "hex"), msg);
p.decrypt(Buffer.from(k.privateKey, "hex"), ct);
console.log("captured AES key len:", aesKey.length);

const raw = Buffer.from(ct.startsWith("encrypted:") ? ct.slice(10) : ct, "base64");
const eph = raw.subarray(0, 65);
const nonce = raw.subarray(65, 81);
const rest = raw.subarray(81);
const tag = rest.subarray(rest.length - 16);
const data = rest.subarray(0, rest.length - 16);

// full shared point via BigInt EC math (same curve ops as app.config)
const x = BigInt("0x" + eph.subarray(1, 33).toString("hex"));
const y = BigInt("0x" + eph.subarray(33, 65).toString("hex"));
const P = 2n ** 256n - 2n ** 32n - 977n;
const priv = BigInt("0x" + k.privateKey);
function mod(a, m) { const r = a % m; return r >= 0n ? r : r + m; }
function inv(a, m) {
  // extended Euclid — modpow with a 256-bit modulus overflows node's BigInt
  let [old_r, r] = [mod(a, m), m];
  let [old_s, s] = [1n, 0n];
  while (r !== 0n) {
    const q = old_r / r;
    [old_r, r] = [r, old_r - q * r];
    [old_s, s] = [s, old_s - q * s];
  }
  return mod(old_s, m);
}
function pointAdd(p1, p2) {
  if (p1 === null) return p2;
  if (p1[0] === p2[0] && mod(p1[1] + p2[1], P) === 0n) return null;
  let lam;
  const samePoint = p1[0] === p2[0] && p1[1] === p2[1];
  if (samePoint) {
    lam = mod(3n * p1[0] * p1[0] * inv(2n * p1[1], P), P);
  } else {
    lam = mod((p2[1] - p1[1]) * inv(p2[0] - p1[0], P), P);
  }
  const x3 = mod(lam * lam - p1[0] - p2[0], P);
  return [x3, mod(lam * (p1[0] - x3) - p1[1], P)];
}
let pt = null, addend = [x, y], s = priv;
while (s > 0n) {
  if (s & 1n) pt = pointAdd(pt, addend);
  addend = pointAdd(addend, addend);
  s >>= 1n;
}
const sharedX = pt[0], sharedY = pt[1];
const hex = (n) => Buffer.from(n.toString(16).padStart(64, "0"), "hex");
const X32 = hex(sharedX);
const full = Buffer.concat([Buffer.from([4]), X32, hex(sharedY)]);
const compressed = Buffer.concat([Buffer.from([sharedY % 2n === 0n ? 2 : 3]), X32]);
const ephXY = eph.subarray(1);

const hk = (ikm) => Buffer.from(crypto.hkdfSync("sha256", ikm, Buffer.alloc(0), Buffer.alloc(0), 32));
const cands = {
  "eph+XY64 (current app)": Buffer.concat([eph, ephXY]),
  "eph+X32": Buffer.concat([eph, X32]),
  "eph+full65": Buffer.concat([eph, full]),
  "eph+compressed33": Buffer.concat([eph, compressed]),
  "full65 alone": full,
  "X32 alone": X32,
};
// sanity: our captured key must actually decrypt
try {
  const d = crypto.createDecipheriv.call(crypto, "aes-256-gcm", aesKey, nonce);
  d.setAuthTag(tag);
  const out = Buffer.concat([d.update(data), d.final()]);
  console.log("sanity decrypt:", out.toString());
} catch (e) { console.log("sanity decrypt failed:", e.message.slice(0, 40)); }
// note: monkey-patch replaced createDecipheriv on the crypto object; restore
crypto.createDecipheriv = orig;

for (const [name, ikm] of Object.entries(cands)) {
  console.log(name.padEnd(24), "len", ikm.length, hk(ikm).equals(aesKey) ? "MATCH" : "no");
}
