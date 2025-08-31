#!/usr/bin/env python3
# get_channel_ids.py — resolve Telegram chat IDs & admin status for your bot
import os, re, sys, requests

API = "https://api.telegram.org/bot{}/{}"

def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(1)

def norm_ident(x: str) -> str:
    x = x.strip()
    m = re.match(r"https?://t\.me/(.+)$", x, re.I)   # allow t.me links
    if m: x = m.group(1)
    if not x.startswith("-100") and not x.startswith("@"):
        x = "@" + x
    return x

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        die("TELEGRAM_BOT_TOKEN not set (export it first).")

    inputs = sys.argv[1:]
    if not inputs:
        for k in ("TELEGRAM_CHAT_FREE", "TELEGRAM_CHAT_BASIC", "TELEGRAM_CHAT_PRO", "TELEGRAM_CHAT_VIP"):
            v = os.getenv(k)
            if v: inputs.append(v)
    if not inputs:
        die("Provide at least one @channel / t.me/... / numeric -100… ID.")

    # Get bot info
    me = requests.get(API.format(token, "getMe"), timeout=20).json()
    if not me.get("ok"):
        die(f"getMe failed: {me}")
    bot_id = me["result"]["id"]
    bot_username = me["result"]["username"]

    for raw in inputs:
        ident = norm_ident(raw)
        print(f"\n=== {raw} ===")
        chat = requests.get(API.format(token, "getChat"), params={"chat_id": ident}, timeout=20).json()
        if not chat.get("ok"):
            print("getChat ->", chat)
            continue

        result = chat["result"]
        cid = result["id"]
        title = result.get("title") or result.get("username") or "(no title)"
        ctype = result.get("type")

        print(f"id: {cid}")
        print(f"title: {title}")
        print(f"type: {ctype}")

        try:
            gm = requests.get(API.format(token, "getChatMember"),
                              params={"chat_id": cid, "user_id": bot_id},
                              timeout=20).json()
            if gm.get("ok"):
                status = gm["result"].get("status")
                is_admin = status in ("administrator", "creator")
                print(f"bot: @{bot_username} status='{status}' is_admin={is_admin}")
            else:
                print("getChatMember ->", gm)
        except Exception as e:
            print("getChatMember exception ->", e)

if __name__ == "__main__":
    main()
