#!/usr/bin/env bash
# Safe disk cleanup — the same routine as the panel's "Clean up" button.
#
# Cron (this container has no cron, your host probably does):
#   15 4 * * *  cd /home/daytona/codebase && .venv/bin/python tools/disk_cleanup.py >> data/logs/disk_cleanup.log 2>&1
# or use the systemd timer in tools/systemd/.
#
# Nothing here touches .env, cookies.txt, the job documents, the DB backup,
# the venv, node_modules, fonts, emoji assets, or files a running batch holds.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python tools/disk_cleanup.py "$@"
