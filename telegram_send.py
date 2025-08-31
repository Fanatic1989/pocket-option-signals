import os, requests

MAX_CHARS = 3800  # headroom below Telegram 4096 hard cap

def _send_plain(text: str, chat_id: str):
    bot = os.getenv("TELEGRAM_BOT_TOKEN")
    if not (bot and chat_id):
        print(f"⚠️ send skipped: missing bot or chat_id ({chat_id})")
        return False
        # Coerce to string to avoid TypeError when someone passes bool/None/etc.
    if not isinstance(text, str):
        text = str(text or "")
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

def send_to_tiers(msg: str):
    # suppress empty messages entirely
    if not isinstance(msg, str):
        msg = str(msg or "")
    if not msg.strip() or msg.strip() == "(empty)":
        print("✋ suppressed empty (no confirmed signals).")
        return False
    """Broadcast to all tier chats (FREE, BASIC, PRO, VIP)."""
    if not isinstance(msg, str):
        msg = str(msg or "")
    any_ok = False
    for env_k in ("TELEGRAM_CHAT_FREE","TELEGRAM_CHAT_BASIC","TELEGRAM_CHAT_PRO","TELEGRAM_CHAT_VIP"):
        cid = os.getenv(env_k, "")
        if not cid: 
            print(f"{env_k} empty; skip")
            continue
        if _send_plain(msg, cid):
            any_ok = True
            print(f"✅ Sent to {env_k}")
        else:
            print(f"❌ Failed to send to {env_k}")
    return any_ok

def send_confirmed(text: str):
    """
    Keep header (first 2–3 lines) + only confirmed entries:
    lines starting with 🟢/🔴 and their single “• reason” line.
    If none and SUPPRESS_EMPTY=1 (default) → skip send.
    """
    SUP = os.getenv("SUPPRESS_EMPTY","1") == "1"
    lines = text.splitlines()
    keep, next_is_reason = [], False

    for i, ln in enumerate(lines):
        if i < 3:
            keep.append(ln); continue
        s = ln.lstrip()
        if s.startswith("🟢") or s.startswith("🔴"):
            keep.append(ln); next_is_reason = True; continue
        if next_is_reason and s.startswith("•"):
            keep.append(ln); keep.append(""); next_is_reason = False; continue

    has_confirmed = any(l.lstrip().startswith(("🟢","🔴")) for l in keep[3:])
    msg = "\n".join(keep).strip()
    if has_confirmed or not SUP:
        return send_to_tiers(msg)
    else:
        print("✋ suppressed empty (no confirmed signals).")
        return False
