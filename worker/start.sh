#!/usr/bin/env bash
# Re-run under bash if invoked as `sh start.sh`: a bare "Illegal option -o
# pipefail" is not a useful error to hand someone on a phone.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

# Clipforge — run the whole phone side with one command.
#
#     ./start.sh                                  # worker + cloudflared + auto-register
#     APP_URL="https://your-app-url" ./start.sh
#     ./start.sh --tunnel ngrok                   # ngrok instead of cloudflared
#     ./start.sh --no-tunnel                      # worker only (local testing)
#
# It starts, and keeps alive:
#   1. the download worker   node start.js        on 127.0.0.1:$WORKER_PORT
#   2. a tunnel              cloudflared / ngrok  to that port
#   3. register.sh           which tells the app the tunnel link, and tells it
#                            again every time cloudflared hands out a new one
#
# Needs: node, yt-dlp, curl, and cloudflared or ngrok.
# The token comes from ./token.txt (chmod 600) or $WORKER_TOKEN, and must match
# the app's WORKER_TOKEN. It is never printed — only its last 4 characters.
#
# Options:
#   --app URL          app base URL                  (default: $APP_URL)
#   --port N           worker port                   (default: 3001)
#   --tunnel NAME      cloudflared | ngrok | auto    (default: auto)
#   --no-tunnel        skip the tunnel (and register.sh)
#   --no-register      tunnel, but don't auto-register
#   --help             this text
#
# Logs: worker.log, cloudflared.log (or ngrok.log), register.log

set -uo pipefail
# Absolute path before the cd below, so --help can still read this file.
SELF=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
cd "$(dirname -- "$SELF")"

APP_URL=${APP_URL:-}
WORKER_PORT=${WORKER_PORT:-3001}
# start.js reads WORKER_PORT from the environment — export it or the worker binds
# 3001 while the tunnel and health check point at --port.
export WORKER_PORT
TUNNEL=auto
WITH_TUNNEL=1
WITH_REGISTER=1

while [ $# -gt 0 ]; do
  case $1 in
    --app) APP_URL=${2:-}; shift 2 ;;
    --app=*) APP_URL=${1#*=}; shift ;;
    --port) WORKER_PORT=${2:-3001}; shift 2 ;;
    --port=*) WORKER_PORT=${1#*=}; shift ;;
    --tunnel) TUNNEL=${2:-auto}; shift 2 ;;
    --tunnel=*) TUNNEL=${1#*=}; shift ;;
    --no-tunnel) WITH_TUNNEL=0; WITH_REGISTER=0; shift ;;
    --no-register) WITH_REGISTER=0; shift ;;
    -h|--help) sed -n '/^# Clipforge —/,/^# Logs:/p' "$SELF" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "start.sh: unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'start.sh: %s\n' "$*" >&2; exit 1; }

# ---- settings --------------------------------------------------------------
TOKEN=${WORKER_TOKEN:-}
if [ -z "$TOKEN" ] && [ -f token.txt ]; then
  TOKEN=$(tr -d ' \t\r\n' < token.txt)
fi
[ -n "$TOKEN" ] ||
  die "no worker token. Write it to token.txt (chmod 600) or set WORKER_TOKEN — it must match the app's WORKER_TOKEN."

if [ "$WITH_REGISTER" = 1 ]; then
  [ -n "$APP_URL" ] || die "no app URL. Set APP_URL=https://... or pass --app URL (or use --no-register)."
  case $APP_URL in
    http://*|https://*) ;;
    *) die "the app URL must start with http:// or https:// (got $APP_URL)" ;;
  esac
  APP_URL=${APP_URL%/}
fi

for bin in node yt-dlp curl; do
  command -v "$bin" >/dev/null 2>&1 ||
    die "$bin is required (Termux: pkg install nodejs-lts yt-dlp curl)"
done

TUNNEL_LOG=cloudflared.log
if [ "$WITH_TUNNEL" = 1 ]; then
  case $TUNNEL in
    auto)
      if command -v cloudflared >/dev/null 2>&1; then
        TUNNEL=cloudflared
      elif command -v ngrok >/dev/null 2>&1; then
        TUNNEL=ngrok
      else
        die "needs cloudflared or ngrok (Termux: pkg install cloudflared), or run with --no-tunnel"
      fi
      ;;
    cloudflared|ngrok)
      command -v "$TUNNEL" >/dev/null 2>&1 || die "$TUNNEL is not installed"
      ;;
    *) die "--tunnel must be cloudflared, ngrok or auto (got $TUNNEL)" ;;
  esac
  [ "$TUNNEL" = ngrok ] && TUNNEL_LOG=ngrok.log
fi

masked=****
[ ${#TOKEN} -ge 4 ] && masked="****${TOKEN: -4}"

# ---- lifecycle -------------------------------------------------------------
WORKER_PID=
TUNNEL_PID=
REG_PID=

# Children first: restarting a service replaces the PID we track, and a dead
# parent can leave an orphan holding the tunnel or the port.
kill_tree() {
  pid=$1
  [ -z "$pid" ] && return 0
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child"; done
  kill "$pid" 2>/dev/null
}

cleanup() {
  trap - INT TERM EXIT
  say ""
  say "==> stopping"
  for pid in "$WORKER_PID" "$TUNNEL_PID" "$REG_PID"; do kill_tree "$pid"; done
  sleep 1
  for pid in "$WORKER_PID" "$TUNNEL_PID" "$REG_PID"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
  done
  command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock
}
trap cleanup INT TERM EXIT

start_worker() {
  WORKER_TOKEN="$TOKEN" node start.js >>worker.log 2>&1 &
  WORKER_PID=$!
}

start_tunnel() {
  : >"$TUNNEL_LOG" # fresh log: a stale link is never re-registered
  if [ "$TUNNEL" = cloudflared ]; then
    cloudflared tunnel --url "http://127.0.0.1:$WORKER_PORT" --no-autoupdate >>"$TUNNEL_LOG" 2>&1 &
  else
    ngrok http "$WORKER_PORT" --log stdout --log-format logfmt >>"$TUNNEL_LOG" 2>&1 &
  fi
  TUNNEL_PID=$!
}

start_register() {
  WORKER_TOKEN="$TOKEN" sh register.sh --app "$APP_URL" --log "$TUNNEL_LOG" >>register.log 2>&1 &
  REG_PID=$!
}

wait_health() {
  tries=0
  while [ "$tries" -lt 20 ]; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
      -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$WORKER_PORT/health") || code=000
    if [ "$code" = "200" ]; then
      say "    GET /health -> 200"
      return 0
    fi
    kill -0 "$WORKER_PID" 2>/dev/null || return 1
    tries=$((tries + 1))
    sleep 1
  done
  return 1
}

show_registration() {
  tries=0
  while [ "$tries" -lt 25 ]; do
    line=$(grep -m1 'saved https' register.log 2>/dev/null)
    if [ -n "$line" ]; then
      say "    $line"
      return 0
    fi
    if grep -qE 'rejected|no answer|no token|no tunnel link' register.log 2>/dev/null; then
      say "    $(grep -E 'rejected|no answer|no token|no tunnel link' register.log | tail -n 1)"
      return 1
    fi
    tries=$((tries + 1))
    sleep 1
  done
  say "    still waiting for the tunnel link — see register.log"
}

# ---- boot ------------------------------------------------------------------
say "==> Clipforge worker on Termux"
say "    worker  127.0.0.1:$WORKER_PORT   token $masked"
[ "$WITH_TUNNEL" = 1 ] && say "    tunnel  $TUNNEL -> $TUNNEL_LOG"
[ "$WITH_REGISTER" = 1 ] && say "    app     $APP_URL (auto-register)"
say ""

command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock

start_worker
if wait_health; then
  :
else
  say "    the worker did not answer /health yet — check worker.log:"
  tail -n 5 worker.log 2>/dev/null | sed 's/^/      /'
fi

if [ "$WITH_TUNNEL" = 1 ]; then
  start_tunnel
  say "    waiting for the tunnel link…"
  if [ "$WITH_REGISTER" = 1 ]; then
    start_register
    show_registration
  fi
fi

say ""
say "==> running. Ctrl-C stops the worker, the tunnel and register.sh."
say ""

while :; do
  sleep 5
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    say "worker stopped — restarting (see worker.log)"
    start_worker
  fi
  if [ "$WITH_TUNNEL" = 1 ] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    say "tunnel stopped — restarting; the new link is re-registered automatically"
    sleep 2
    start_tunnel
  fi
  if [ "$WITH_REGISTER" = 1 ] && ! kill -0 "$REG_PID" 2>/dev/null; then
    say "register.sh stopped — restarting"
    sleep 2
    start_register
  fi
done
