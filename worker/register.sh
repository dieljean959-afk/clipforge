#!/bin/sh
# Clipmake worker auto-registration.
#
# cloudflared/ngrok hand out a new https link every time they restart, and the
# app can only reach the phone through that link. This script reads the current
# link out of the tunnel log and posts it to the app's register endpoint, so the
# link is never pasted into Settings by hand.
#
# One line in your Termux start.sh (cloudflared logging to ./cloudflared.log):
#
#     APP_URL="https://your-app-url" sh register.sh >register.log 2>&1 &
#
# Run cloudflared so it writes that log:
#
#     cloudflared tunnel --url http://127.0.0.1:8765 --no-autoupdate 2>&1 | tee cloudflared.log
#
# Options (all optional):
#   --app URL         app base URL                 (default: $APP_URL)
#   --token TOKEN     worker token                 (default: $WORKER_TOKEN)
#   --token-file PATH file holding the token       (default: ./token.txt)
#   --url URL         register this URL, skip log detection
#   --log PATH        tunnel log to watch          (default: ./cloudflared.log)
#   --interval SEC    log check interval           (default: 5)
#   --once            register once, then exit
#   --quiet           only print registrations and errors
#
# Exit codes: 0 registered, 1 the app rejected it, 2 bad local setup.
# The token is never printed — only its last 4 characters.

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

APP_URL=${APP_URL:-}
TOKEN=${WORKER_TOKEN:-}
TOKEN_FILE="$SCRIPT_DIR/token.txt"
LOG="$SCRIPT_DIR/cloudflared.log"
URL_OVERRIDE=
INTERVAL=5
ONCE=0
QUIET=0

while [ $# -gt 0 ]; do
  case $1 in
    --app) APP_URL=${2:-}; shift 2 ;;
    --app=*) APP_URL=${1#*=}; shift ;;
    --token) TOKEN=${2:-}; shift 2 ;;
    --token=*) TOKEN=${1#*=}; shift ;;
    --token-file) TOKEN_FILE=${2:-}; shift 2 ;;
    --token-file=*) TOKEN_FILE=${1#*=}; shift ;;
    --url) URL_OVERRIDE=${2:-}; shift 2 ;;
    --url=*) URL_OVERRIDE=${1#*=}; shift ;;
    --log) LOG=${2:-}; shift 2 ;;
    --log=*) LOG=${1#*=}; shift ;;
    --interval) INTERVAL=${2:-5}; shift 2 ;;
    --interval=*) INTERVAL=${1#*=}; shift ;;
    --once) ONCE=1; shift ;;
    --quiet|-q) QUIET=1; shift ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "register.sh: unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { [ "$QUIET" = 1 ] && return 0; echo "register.sh: $*"; }
die() { echo "register.sh: $*" >&2; exit 2; }

case $APP_URL in
  http://*|https://*) ;;
  "") die "no app URL. Set APP_URL=... or pass --app URL." ;;
  *) die "the app URL must start with http:// or https:// (got $APP_URL)" ;;
esac
APP_URL=${APP_URL%/}

if [ -z "$TOKEN" ]; then
  [ -f "$TOKEN_FILE" ] ||
    die "no token: $TOKEN_FILE is missing. Write the worker token there (chmod 600) or set WORKER_TOKEN."
  TOKEN=$(tr -d ' \t\r\n' <"$TOKEN_FILE")
fi
[ -n "$TOKEN" ] || die "the token in $TOKEN_FILE is empty."
case $TOKEN in
  *[!0-9A-Za-z._~+/=-]*) die "the token contains characters a worker token never has — check $TOKEN_FILE." ;;
esac

masked=****
[ ${#TOKEN} -ge 4 ] && masked="****${TOKEN#"${TOKEN%????}"}"

tmp=$(mktemp 2>/dev/null || echo "${TMPDIR:-/tmp}/clipmake-register.$$")
trap 'rm -f "$tmp" "$tmp.err"' EXIT
trap 'exit 0' INT TERM

find_url() {
  [ -r "$LOG" ] || return 1
  grep -oE 'https://[A-Za-z0-9._-]+\.(trycloudflare\.com|ngrok[a-z.-]*)' "$LOG" 2>/dev/null | tail -n 1
}

json_field() { # json_field <field> <file>
  sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$2" 2>/dev/null | head -n 1
}

# Returns 0 = saved, 1 = the app rejected this link/token, 2 = retry later.
register() {
  url=$1
  case $url in
    https://*) ;;
    *) echo "register.sh: refusing $url — a tunnel link must be https" >&2; return 1 ;;
  esac
  code=$(curl -sS -o "$tmp" -w '%{http_code}' --max-time 25 \
    -X POST "$APP_URL/api/hosting/worker/register" \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' \
    --data-binary "{\"url\":\"$url\"}" 2>"$tmp.err") || code=000

  case $code in
    200)
      msg=$(json_field message "$tmp")
      echo "register.sh: saved $url (HTTP 200)${msg:+ — $msg}"
      return 0
      ;;
    401)
      echo "register.sh: the app rejected the token (HTTP 401)." >&2
      echo "  token.txt must hold the same value as the app's WORKER_TOKEN." >&2
      echo "  Settings → Download worker should show ...$masked" >&2
      return 1
      ;;
    400|422)
      echo "register.sh: the app rejected the link (HTTP $code): $(json_field detail "$tmp")" >&2
      return 1
      ;;
    503)
      echo "register.sh: the app has no worker token saved yet (HTTP 503): $(json_field detail "$tmp")" >&2
      return 1
      ;;
    *)
      detail=$(json_field detail "$tmp")
      err=$(tr -d '\r\n' <"$tmp.err" 2>/dev/null)
      echo "register.sh: no answer from $APP_URL (HTTP $code)${detail:+ — $detail}${err:+ — $err}" >&2
      return 2
      ;;
  esac
}

say "app $APP_URL, token $masked"
[ -n "$URL_OVERRIDE" ] || { [ "$QUIET" = 1 ] || say "watching $LOG"; }

last_url=
delay=0
while :; do
  if [ -n "$URL_OVERRIDE" ]; then
    url=$URL_OVERRIDE
  else
    url=$(find_url)
  fi

  if [ -z "$url" ]; then
    [ "$ONCE" = 1 ] && die "no tunnel link in $LOG yet — run cloudflared with: 2>&1 | tee $LOG"
    say "waiting for a tunnel link in $LOG"
    sleep "$INTERVAL"
    continue
  fi

  if [ "$url" != "$last_url" ] || [ "$delay" -gt 0 ]; then
    register "$url"
    case $? in
      0)
        last_url=$url
        delay=0
        [ -n "$URL_OVERRIDE" ] && exit 0
        ;;
      1)
        # A rejection will not fix itself; wait for a NEW link instead of looping.
        last_url=$url
        delay=0
        [ "$ONCE" = 1 ] && exit 1
        [ -n "$URL_OVERRIDE" ] && exit 1
        ;;
      *)
        [ "$ONCE" = 1 ] && exit 1
        delay=$((delay + 5))
        [ "$delay" -gt 30 ] && delay=30
        say "retrying in ${delay}s"
        sleep "$delay"
        continue
        ;;
    esac
  fi

  [ "$ONCE" = 1 ] && exit 0
  sleep "$INTERVAL"
done
