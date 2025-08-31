#!/usr/bin/env bash
set -euo pipefail
attempt=1
for delay in 10 20 40; do
  echo "::group::Attempt $attempt running signals"
  if python run_and_send.py; then
    echo "::endgroup::Success on attempt $attempt"
    exit 0
  fi
  echo "::endgroup::Attempt $attempt failed; retrying in ${delay}s..."
  sleep "$delay"
  attempt=$((attempt+1))
done
echo "All retries failed"
exit 1
