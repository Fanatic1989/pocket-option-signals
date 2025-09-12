# routes.py — compatibility endpoints for your GitHub dashboard
from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
from live_engine import ENGINE, tg_test_all, TIER_TO_CHAT, DAILY_CAPS

bp = Blueprint("dashboard", __name__)  # <-- name matches your template url_for(...)

@bp.route("/")
def home():
    # Your GitHub dashboard renders its own template; keep the file you already have.
    # If your app factory renders a different template, point it there.
    return jsonify({"ok": True, "msg": "Dashboard HTML is served by your existing app template."})

# -------- live controls (your JS calls these) --------
@bp.route("/live/start")
def live_start():
    ok, msg = ENGINE.start()
    return jsonify({"ok": ok, "message": msg, "status": {"state": "RUNNING" if ok else "STOPPED"}})

@bp.route("/live/stop")
def live_stop():
    ok, msg = ENGINE.stop()
    return jsonify({"ok": ok, "message": msg, "status": {"state": "STOPPED"}})

@bp.route("/live/debug/on")
def live_dbg_on():
    ENGINE.set_debug(True)
    return jsonify({"ok": True, "status": {"debug": True}})

@bp.route("/live/debug/off")
def live_dbg_off():
    ENGINE.set_debug(False)
    return jsonify({"ok": True, "status": {"debug": False}})

@bp.route("/live/status")
def live_status():
    st = ENGINE.status()
    return jsonify({
        "ok": True,
        "status": {"state": "RUNNING" if st["running"] else "STOPPED", "debug": st["debug"], "day": st["tally"]["date"]},
        "detail": st
    })

@bp.route("/live/tally")
def live_tally():
    t = ENGINE.tally()
    by = t["by_tier"]
    return jsonify({"ok": True, "tally": {"free": by["free"], "basic": by["basic"], "pro": by["pro"], "vip": by["vip"], "all": t["total"]}})

# -------- telegram diagnostics + test (your dashboard buttons) --------
@bp.route("/telegram/diag")
def telegram_diag():
    st = ENGINE.status()
    return jsonify({
        "ok": True,
        "configured_tiers": {k: bool(v) for k, v in TIER_TO_CHAT.items()},
        "caps": DAILY_CAPS,
        "last_send_result": st.get("last_send_result"),
    })

@bp.route("/telegram/test", methods=["POST", "GET"])
def telegram_test():
    # Sends 1 short test message to each configured tier WITHOUT using caps
    ok, info_json = tg_test_all()
    return jsonify({"ok": ok, "result": info_json})

# -------- generic send endpoint you can call from forms/tools ---------------
@bp.route("/telegram/send", methods=["POST"])
def telegram_send():
    data = request.get_json(silent=True) or request.form or {}
    tier = (data.get("tier") or "").lower().strip()
    text = data.get("text") or ""
    if tier not in ("free", "basic", "pro", "vip"):
        return jsonify({"ok": False, "error": "tier must be free/basic/pro/vip"}), 400
    if not text:
        return jsonify({"ok": False, "error": "text required"}), 400
    ok, msg = ENGINE.send_signal(tier, text)
    return jsonify({"ok": ok, "message": msg, "status": ENGINE.status()})

# -------- health --------
@bp.route("/_up")
def up():
    return jsonify({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})
