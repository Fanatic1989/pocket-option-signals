# app.py — WSGI entrypoint for gunicorn: exports `app`
import os
from flask import Flask, jsonify, request
from routes import bp as dashboard_bp
from live_engine import ENGINE, tg_test_all, TIER_TO_CHAT, DAILY_CAPS

def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    # Secret key for sessions / flashing
    app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

    # Register dashboard (all UI, backtest, users, /live/* and /telegram/* routes)
    app.register_blueprint(dashboard_bp, url_prefix="")

    # ---- Minimal JSON API (handy for alt UIs / probes) ----------------------
    @app.get("/api/status")
    def api_status():
        return jsonify(ENGINE.status())

    @app.post("/api/send")
    def api_send():
        data = request.get_json(silent=True) or {}
        tier = (data.get("tier") or "vip").lower()
        text = data.get("text") or ""
        res = ENGINE.send_to_tier(tier, text)
        return jsonify({"result": res, "status": ENGINE.status()})

    @app.post("/api/test/vip")
    def api_test_vip():
        data = request.get_json(silent=True) or {}
        text = data.get("text") or "🧪 VIP test from API"
        res = ENGINE.send_to_tier("vip", text)
        return jsonify({"result": res, "status": ENGINE.status()})

    @app.get("/api/check_bot")
    def api_check_bot():
        # Quick /getMe plus the configured chat flags
        import requests
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        out = {"ok": False, "configured_chats": {k: bool(v) for k, v in TIER_TO_CHAT.items()}}
        if token:
            try:
                r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
                out["getMe"] = r.json()
                out["ok"] = bool(out["getMe"].get("ok"))
            except Exception as e:
                out["getMe"] = {"ok": False, "error": str(e)}
        return jsonify(out)

    @app.post("/api/start")
    def api_start():
        ok, msg = ENGINE.start()
        return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

    @app.post("/api/stop")
    def api_stop():
        ok, msg = ENGINE.stop()
        return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

    @app.post("/api/debug_on")
    def api_dbg_on():
        ENGINE.set_debug(True)
        return jsonify({"ok": True, "status": ENGINE.status()})

    @app.post("/api/debug_off")
    def api_dbg_off():
        ENGINE.set_debug(False)
        return jsonify({"ok": True, "status": ENGINE.status()})

    # Health
    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "caps": DAILY_CAPS})

    return app

# Gunicorn entrypoint
app = create_app()
