import os, requests

def send_telegram(text: str, chat_id: str):
    """Send plain-text to Telegram in safe chunks (no Markdown)."""
    bot = os.getenv("TELEGRAM_BOT_TOKEN")
    if not (bot and chat_id):
        print(f"⚠️ send_telegram skipped: missing bot or chat_id for {chat_id}")
        return
    url = f"https://api.telegram.org/bot{bot}/sendMessage"
    MAX = 3800  # below Telegram ~4096 limit to be safe
    parts = [text[i:i+MAX] for i in range(0, len(text), MAX)] or ["(empty)"]
    ok_any = False
    for i, part in enumerate(parts, 1):
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": part}, timeout=30)
            js = r.json()
        except Exception as e:
            js = {"ok": False, "error": str(e)}
        if js.get("ok"):
            ok_any = True
        else:
            print(f"❌ send_telegram failed ({i}/{len(parts)}):", js)
    if not ok_any:
        print("⚠️ No successful sends for", chat_id)

def send_to_tiers(text: str):
    """Broadcast to all tier channels if set."""
    for envkey in ("TELEGRAM_CHAT_FREE","TELEGRAM_CHAT_BASIC","TELEGRAM_CHAT_PRO","TELEGRAM_CHAT_VIP"):
        cid = os.getenv(envkey, "")
        if cid:
            send_telegram(text, cid)
            print(f"✅ Sent to {envkey}")
        else:
            print(f"⚠️ {envkey} empty; skipped")
