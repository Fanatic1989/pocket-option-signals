import os, requests

MAX_CHARS = 3800  # headroom below Telegram 4096 cap

def _send_plain(text: str, chat_id: str) -> bool:
    bot = os.getenv("TELEGRAM_BOT_TOKEN")
    if not (bot and chat_id):
        print(f"⚠️ send skipped: missing bot or chat_id ({chat_id})")
        return False
    url = f"https://api.telegram.org/bot{bot}/sendMessage"
    chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)] or ["(empty)"]
    ok = True
    for idx, part in enumerate(chunks, 1):
        try:
            r = requests.post(url, data={"chat_id": chat_id, "text": part}, timeout=30)
            js = r.json()
            if not js.get("ok"):
                ok = False
                print(f"send chunk {idx}/{len(chunks)} failed -> {js}")
        except Exception as e:
            ok = False
            print(f"send chunk {idx}/{len(chunks)} exception -> {e}")
    return ok

def send_to_tiers(msg: str, only: str | None = None):
    """Broadcast to all tier chats (FREE, BASIC, PRO, VIP)."""
    any_ok = False
    tiers = [("FREE","TELEGRAM_CHAT_FREE"),("BASIC","TELEGRAM_CHAT_BASIC"),("PRO","TELEGRAM_CHAT_PRO"),("VIP","TELEGRAM_CHAT_VIP")]
    delivered = 0
    for name, env_k in tiers:
        cid = os.getenv(env_k, "")
        if only and name != only:
            continue
        if not cid: 
            print(f"ℹ️ {env_k} empty; skipping.")
            continue
        if _send_plain(msg, cid):
            delivered += 1
            any_ok = True
            print(f"✅ Sent to {env_k}")
    if not any_ok:
        print("❌ Nothing sent (no chats or all failed).")

def send_confirmed(text: str) -> str | None:
    """
    Keep header + only 🟢/🔴 lines (and the next reason bullet).
    Return filtered string, or None if nothing confirmed and SUPPRESS_EMPTY=1.
    """
    SUP = os.getenv("SUPPRESS_EMPTY", "1") == "1"
    lines = text.splitlines()
    keep, next_is_reason = [], False

    for i, ln in enumerate(lines):
        if i < 2:  # keep header (first two lines)
            keep.append(ln); 
            continue
        s = ln.lstrip()
        if s.startswith("🟢") or s.startswith("🔴"):
            keep.append(ln)
            next_is_reason = True
            continue
        if next_is_reason and s.startswith("•"):
            keep.append(ln)
            keep.append("")  # spacer
            next_is_reason = False
            continue

    body_has_signal = any(ln.lstrip().startswith(("🟢","🔴")) for ln in keep[2:])
    if not body_has_signal and SUP:
        return None
    return "\n".join(keep).strip()
