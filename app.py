# app.py — WSGI entrypoint for gunicorn: exports `app`
import os
from flask import Flask, jsonify
from routes import bp as dashboard_bp
# IMPORTANT: do NOT import tg_test_all (it doesn't exist). Keep this minimal.
from live_engine import ENGINE, TIER_TO_CHAT, DAILY_CAPS

def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Secret key for sessions / flashing (supports both env names)
    app.secret_key = (
        os.getenv("SECRET_KEY")
        or os.getenv("FLASK_SECRET_KEY")
        or "dev-secret-change-me"
    )

    # Register all UI + APIs from the blueprint (routes.py defines them)
    app.register_blueprint(dashboard_bp, url_prefix="")

    # Lightweight health endpoint (unique path; no collision with routes.py)
    @app.get("/healthz")
    def healthz():
        return jsonify({
            "ok": True,
            "caps": DAILY_CAPS,
            "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
            "engine": ENGINE.status(),
        })

    return app

# Gunicorn entrypoint
app = create_app()
