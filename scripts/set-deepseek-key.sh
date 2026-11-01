#!/usr/bin/env bash
# Replace / rotate the DeepSeek API key WITHOUT echoing it.
#
#   printf '%s' 'sk-your-new-key' | ./scripts/set-deepseek-key.sh
#   ./scripts/set-deepseek-key.sh            # prompts with hidden input
#
# The key is never printed, never passed on a command line (so it stays out of
# `ps` and shell history) and never written to a log. Only a masked confirmation
# is shown.
#
# Get the new key from https://platform.deepseek.com → API keys → Create.
set -uo pipefail
cd "$(dirname "$0")/.."

ENVFILE="${1:-.env.local}"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ -t 0 ]; then
  read -r -s -p "Paste the new DeepSeek key (input hidden): " KEY
  echo
else
  IFS= read -r KEY
fi

# strip CR (pasted from Windows/phone) and surrounding whitespace
KEY="${KEY%$'\r'}"
KEY="${KEY#"${KEY%%[![:space:]]*}"}"
KEY="${KEY%"${KEY##*[![:space:]]}"}"

case "$KEY" in
  sk-*) ;;
  *) echo "Refusing: a DeepSeek key starts with 'sk-'." >&2; exit 2 ;;
esac
if [ "${#KEY}" -lt 20 ]; then
  echo "Refusing: that is too short to be a real key." >&2
  exit 2
fi
case "$KEY" in
  *[[:space:]]*) echo "Refusing: the key contains whitespace." >&2; exit 2 ;;
esac

KEY="$KEY" ENVFILE="$ENVFILE" "$PY" - <<'PY'
import os, pathlib, re

key, path = os.environ["KEY"], pathlib.Path(os.environ["ENVFILE"])
lines = path.read_text().splitlines() if path.exists() else []
out, replaced = [], False
for line in lines:
    if re.match(r"^\s*DEEPSEEK_API_KEY\s*=", line):
        out.append(f"DEEPSEEK_API_KEY={key}")
        replaced = True
    else:
        out.append(line)
if not replaced:
    out.append(f"DEEPSEEK_API_KEY={key}")

tmp = path.with_name(path.name + ".tmp")
tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
tmp.chmod(0o600)
tmp.replace(path)
print(f"{'updated' if replaced else 'added'} DEEPSEEK_API_KEY in {path}")
print(f"masked: {key[:4]}…{key[-4:]}  (len {len(key)})")
PY

# The backend re-resolves provider keys on every request, so this is enough to
# pick the new value up — no restart needed.
if command -v curl >/dev/null 2>&1; then
  if curl -sS -o /dev/null --max-time 5 http://127.0.0.1:8000/api/health 2>/dev/null; then
    curl -sS --max-time 8 http://127.0.0.1:8000/api/health 2>/dev/null \
      | "$PY" -c 'import json,sys
try:
    d = json.load(sys.stdin)["llmDeepSeek"]
    print("deepseek key loaded:", d.get("keyLoaded"), "| source:", d.get("source"))
except Exception:
    print("key applied; health check unavailable")'
  else
    echo "key applied (backend not running — it will pick this up on next start)"
  fi
fi
