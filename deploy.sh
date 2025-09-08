#!/usr/bin/env bash
set -euo pipefail
echo "Pushing to GitHub: Fanatic1989/pocket-option-signals (main)"
git init
git add -A
git commit -m "Deploy modular Pocket Option Signals"
git branch -M main
git remote remove origin >/dev/null 2>&1 || true
git remote add origin https://github.com/Fanatic1989/pocket-option-signals.git
git push -u origin main --force
echo "Done. Now connect the repo on Render and deploy."
