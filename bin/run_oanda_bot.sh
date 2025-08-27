#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$HOME/pocket-option-signals"
ENV_FILE="$HOME/.config/oanda-bot.env"

# Load env
set -a
source "$ENV_FILE"
set +a

# Prefer venv if it exists
if [ -f "$APP_DIR/venv/bin/python3" ]; then
  PY="$APP_DIR/venv/bin/python3"
else
  PY="python3"
fi

cd "$APP_DIR"

# Ensure dependencies exist (idempotent; cheap if already installed)
$PY - <<'PY'
import sys, subprocess as sp
need = ["pandas","pandas_ta","requests"]
try:
    import pandas, pandas_ta, requests  # noqa
except Exception:
    sp.check_call([sys.executable,"-m","pip","install","--break-system-packages","pandas","pandas_ta","requests"])
PY

# Run the bot
exec "$PY" "$APP_DIR/oanda_signal_bot.py"
