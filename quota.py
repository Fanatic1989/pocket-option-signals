import os, json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data"); DATA.mkdir(exist_ok=True)
F = DATA/"vip_quota.json"

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def get_today() -> int:
    if not F.exists(): return 0
    try:
        j = json.loads(F.read_text())
        if j.get("date")==_today(): return int(j.get("count",0))
    except: pass
    return 0

def add(n:int):
    d = _today()
    cur = get_today()
    F.write_text(json.dumps({"date":d,"count":cur+int(n)}))

def reset_if_new_day():
    d = _today()
    if not F.exists():
        F.write_text(json.dumps({"date":d,"count":0})); return
    try:
        j = json.loads(F.read_text())
        if j.get("date")!=d:
            F.write_text(json.dumps({"date":d,"count":0}))
    except:
        F.write_text(json.dumps({"date":d,"count":0}))
