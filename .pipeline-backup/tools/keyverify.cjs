// Official @dotenvx/primitives decrypt — verification probe.
// Key material arrives on stdin (from tools/keyemit.py). Prints ONLY: success
// flag, plaintext length, and whether it looks like a Google API key.
const primitives = require("/tmp/dpx/package/dist/index.cjs");

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", () => {
  const { value, priv } = JSON.parse(input);
  console.log(`value len: ${value.length}, prefix ok: ${value.startsWith("encrypted:")}`);
  console.log(`priv len: ${priv.length}`);
  if (!value || !priv) process.exit(3);

  const privKey = Buffer.from(priv, "hex");
  try {
    const pt = primitives.decrypt(privKey, value);
    const s = Buffer.from(pt).toString("utf8");
    console.log(
      `default curve: SUCCESS, plaintext ${s.length} chars, looksLikeGoogleKey=${s.startsWith("AIza")}`,
    );
    process.exit(0);
  } catch (e) {
    console.log(`default curve: ${e.code || e.constructor.name}: ${String(e.message).slice(0, 100)}`);
  }
  process.exit(1);
});
