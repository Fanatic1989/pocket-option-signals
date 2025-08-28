import os, json, time, requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

DATA = Path("data"); DATA.mkdir(exist_ok=True)
COUNTS_FILE = DATA/"tier_counts.json"
LIMITS_FILE = Path("tier_limits.json")
OFFSET_FILE = DATA/"updates.offset"

def utc_now():
    return datetime.now(timezone.utc)

def tt_now():
    # Trinidad & Tobago UTC-4 (no DST)
    return utc_now() - timedelta(hours=4)

def load_counts():
    today = utc_now().strftime("%Y-%m-%d")
    base = {"date": today, "FREE":0, "BASIC":0, "PRO":0, "VIP":0}
    try:
        j = json.load(open(COUNTS_FILE))
        if j.get("date") != today:
            return base
        for k in base: j.setdefault(k, base[k])
        return j
    except Exception:
        return base

def load_limits():
    try:
        j = json.load(open(LIMITS_FILE))
    except Exception:
        j = {"FREE":3, "BASIC":6, "PRO":15, "VIP":999999}
    # coerce to ints
    for k,v in list(j.items()):
        try: j[k] = int(v)
        except: pass
    return j

def fmt_stats(scope=None):
    counts = load_counts()
    limits = load_limits()
    rows = []
    tiers = ["FREE","BASIC","PRO","VIP"]
    if scope and scope.upper() in tiers:
        tiers = [scope.upper()]
    for t in tiers:
        used = int(counts.get(t,0))
        cap  = int(limits.get(t,0))
        rem  = "∞" if cap>=999999 else max(0, cap-used)
        rows.append(f"{t}: {used}/{('∞' if cap>=999999 else cap)} (remaining {rem})")
    now_tt = tt_now().strftime("%Y-%m-%d %H:%M")
    return "📊 *Today’s usage*\n" + "\n".join(rows) + f"\n\n🕒 TT time: {now_tt}"

def get_offset():
    try:
        return int(OFFSET_FILE.read_text().strip())
    except Exception:
        return None

def save_offset(x):
    OFFSET_FILE.write_text(str(x))

def handle_updates():
    bot = os.environ.get("TELEGRAM_BOT_TOKEN")
    assert bot, "TELEGRAM_BOT_TOKEN missing"
    base = f"https://api.telegram.org/bot{bot}"

    offset = get_offset()
    params = {"timeout": 20}
    if offset is not None:
        params["offset"] = offset + 1

    r = requests.get(f"{base}/getUpdates", params=params, timeout=30)
    j = r.json()
    if not j.get("ok"):
        print("getUpdates error:", j); return

    max_update_id = offset or 0

    for upd in j.get("result", []):
        uid = upd.get("update_id", 0)
        max_update_id = max(max_update_id, uid)

        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        text = (msg.get("text") or "").strip()

        if not text or not chat_id:
            continue

        if text.lower().startswith("/stats"):
            parts = text.split(maxsplit=1)
            scope = None
            if len(parts) == 2:
                arg = parts[1].strip().upper()
                if arg in {"FREE","BASIC","PRO","VIP"}:
                    scope = arg
            reply = fmt_stats(scope)
            # reply plainly (Markdown disabled to be safe)
            requests.post(f"{base}/sendMessage",
                          data={"chat_id": chat_id, "text": reply,
                                "parse_mode": "Markdown",
                                "disable_web_page_preview": True},
                          timeout=20)

        elif text.lower().startswith("/help"):
            help_txt = (
                "🤖 Pocket Signals Bot Commands\n"
                "/stats — show today’s usage for all tiers\n"
                "/stats FREE|BASIC|PRO|VIP — show one tier\n"
                "/help — this help"
            )
            requests.post(f"{base}/sendMessage",
                          data={"chat_id": chat_id, "text": help_txt}, timeout=20)

    save_offset(max_update_id)

if __name__ == "__main__":
    handle_updates()
