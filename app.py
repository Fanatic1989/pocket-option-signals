# app.py — WSGI entrypoint for gunicorn: exports `app`
import os
from flask import Flask, jsonify, request

# --- Live engine (import tolerant, matches routes.py expectations) ---
try:
    from live_engine import ENGINE, TIER_TO_CHAT, DAILY_CAPS
except Exception:
    class _StubEngine:
        def status(self):
            return {
                "running": False, "debug": False, "loop_sleep": 4,
                "tally": {"total": 0, "by_tier": {"free": 0, "basic": 0, "pro": 0, "vip": 0}},
                "tallies": {"all": 0, "free": 0, "basic": 0, "pro": 0, "vip": 0}
            }
        def start(self): return False, "ENGINE not wired"
        def stop(self):  return False, "ENGINE not wired"
        def set_debug(self, _): pass
        def send_to_tier(self, tier, text):  # match routes.py usage
            return {"ok": False, "error": "ENGINE not wired", "tier": tier, "text": text}
    ENGINE = _StubEngine()
    TIER_TO_CHAT = {"free": None, "basic": None, "pro": None, "vip": None}
    DAILY_CAPS = {"free": 3, "basic": 6, "pro": 15, "vip": None}

# --- UI blueprint (dashboard, indicators, backtest, telegram, etc.) ---
from routes import bp as dashboard_bp


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

    # UI + all business endpoints from routes.py
    app.register_blueprint(dashboard_bp, url_prefix="")

    # ------------------------------------------------------------------
    # Minimal “core” JSON API (no overlap with routes.py)
    # NOTE: routes.py already defines /api/status and others;
    # so we keep these on /api/core/* to avoid BuildError collisions.
    # ------------------------------------------------------------------
    @app.get("/api/core/status")
    def core_status():
        return jsonify(ENGINE.status())

    @app.post("/api/core/start")
    def core_start():
        ok, msg = ENGINE.start()
        return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

    @app.post("/api/core/stop")
    def core_stop():
        ok, msg = ENGINE.stop()
        return jsonify({"ok": ok, "msg": msg, "status": ENGINE.status()})

    @app.post("/api/core/debug_on")
    def core_debug_on():
        ENGINE.set_debug(True)
        return jsonify({"ok": True, "status": ENGINE.status()})

    @app.post("/api/core/debug_off")
    def core_debug_off():
        ENGINE.set_debug(False)
        return jsonify({"ok": True, "status": ENGINE.status()})

    @app.post("/api/core/send")
    def core_send():
        """
        Body: {"tier":"vip|pro|basic|free|all", "text":"message"}
        If tier="all", broadcasts to all tiers.
        """
        data = request.get_json(silent=True) or {}
        tier = (data.get("tier") or "vip").lower().strip()
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "Text required"}), 400

        tiers = ["free", "basic", "pro", "vip"] if tier == "all" else [tier]
        results = {t: ENGINE.send_to_tier(t, text) for t in tiers}
        return jsonify({"ok": True, "results": results, "status": ENGINE.status()})

    @app.post("/api/core/test/all")
    def core_test_all():
        msg = "🧪 Core API broadcast test"
        results = {t: ENGINE.send_to_tier(t, f"{msg} ({t.upper()})") for t in ["free", "basic", "pro", "vip"]}
        return jsonify({"ok": True, "results": results})

    @app.get("/api/core/check_bot")
    def core_check_bot():
        # Quick /getMe plus which chats are configured
        import requests
        token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        out = {
            "configured_chats": {k: bool(v) for k, v in (TIER_TO_CHAT or {}).items()},
            "caps": DAILY_CAPS,
            "engine_running": bool(ENGINE.status().get("running")),
        }
        if token:
            try:
                r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
                out["getMe"] = r.json()
                out["ok"] = bool(out["getMe"].get("ok"))
            except Exception as e:
                out["getMe"] = {"ok": False, "error": str(e)}
                out["ok"] = False
        else:
            out["getMe"] = {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}
            out["ok"] = False
        return jsonify(out)

    # Health
    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "caps": DAILY_CAPS})

    return app


# Gunicorn entrypoint
app = create_app()
