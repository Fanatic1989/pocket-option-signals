# --- add these near the other helpers in routes.py ---

def _expand_all_symbols(tokens, cfg):
    """Expand special multi-select tokens into actual symbol lists."""
    out = []
    for t in tokens:
        tt = (t or "").strip().upper()
        if tt in ("ALL", "__ALL__", "__ALL_DERIV__", "ALL_DERIV"):
            out.extend(DERIV_FRX)
        elif tt in ("__ALL_PO__", "ALL_PO", "PO_ALL"):
            out.extend(PO_MAJOR)
        else:
            out.append(t)
    # uniquify while preserving order
    seen, unique = set(), []
    for s in out:
        if s and s not in seen:
            unique.append(s); seen.add(s)
    return unique

def _safe_cfg(x):
    """Ensure config is a dict even if a string sneaks in."""
    from json import loads
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            j = loads(x)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

# --- replace your existing backtest() with this one ---

@bp.route('/backtest', methods=['POST'])
@require_login
def backtest():
    cfg = _safe_cfg(get_config())

    tf = (request.form.get('bt_tf') or 'M5').upper()
    expiry = request.form.get('bt_expiry') or '5m'
    strategy = (request.form.get('bt_strategy') or 'BASE').upper()
    use_server = bool(request.form.get('use_server'))
    app_id = os.getenv("DERV_APP_ID", None) or os.getenv("DERIV_APP_ID", "1089")
    count = int(request.form.get('bt_count') or "300")
    convert_po = bool(request.form.get('convert_po_bt'))

    gran_map = {"M1":60,"M2":120,"M3":180,"M5":300,"M10":600,"M15":900,"M30":1800,"H1":3600,"H4":14400,"D1":86400}
    gran = gran_map.get(tf, 300)

    # -------- symbols resolution (handles ALL / ALL_DERIV / ALL_PO) --------
    raw_syms_text = request.form.get('bt_symbols') or " ".join(cfg.get('symbols') or [])
    text_syms = [s.strip() for s in re.split(r"[,\s]+", raw_syms_text) if s.strip()]
    from_multi = request.form.getlist('bt_symbols_multi')  # may hold ALL tokens
    chosen = from_multi if from_multi else text_syms
    chosen = _expand_all_symbols(chosen, cfg)

    symbols = [_to_deriv(s) if convert_po else s for s in chosen]
    if not symbols:
        session["bt"] = {"error": "No symbols selected. Choose pairs or use ALL/ALL_DERIV/ALL_PO."}
        flash("Backtest error: no symbols selected.")
        return redirect(url_for('dashboard.dashboard'))

    uploaded = request.files.get('bt_csv')
    results = []
    summary = {"trades":0,"wins":0,"losses":0,"draws":0,"winrate":0.0}
    csv_errors = []

    def run_one(sym, df):
        nonlocal summary, results
        # build cfg_run safely each time (avoid any accidental mutation)
        if strategy in ("CUSTOM1","CUSTOM2","CUSTOM3"):
            sid = strategy[-1]
            cfg_run = dict(_safe_cfg(cfg))
            cfg_run["custom"] = _safe_cfg(cfg.get(f"custom{sid}"))
            core = "CUSTOM"
        else:
            cfg_run = dict(_safe_cfg(cfg))
            core = strategy

        # ---- harden call: retry with {} if someone passes a bad cfg downstream ----
        try:
            bt = run_backtest_core_binary(df, core, cfg_run, tf, expiry)
        except Exception as e:
            msg = str(e)
            if "object has no attribute 'get'" in msg or "'get'" in msg:
                bt = run_backtest_core_binary(df, core, {}, tf, expiry)
            else:
                raise

        results.append({
            "symbol": sym,
            "trades": bt.trades,
            "wins": bt.wins,
            "losses": bt.losses,
            "draws": bt.draws,
            "winrate": round(bt.winrate*100,2),
            "rows": bt.rows
        })
        summary["trades"] += bt.trades
        summary["wins"]   += bt.wins
        summary["losses"] += bt.losses
        summary["draws"]  += bt.draws

    try:
        # Single uploaded CSV → run once
        if uploaded and uploaded.filename:
            data = uploaded.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(StringIO(data))
            df.columns = [c.strip().lower() for c in df.columns]
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
            for c in ("open","high","low","close"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
            run_one(symbols[0] if symbols else "CSV", df)
        else:
            # Server CSVs for each symbol
            for sym in symbols:
                try:
                    if use_server:
                        _fetch_one_symbol(app_id, sym, gran, count)
                    path = _deriv_csv_path(sym, gran)
                    if not os.path.exists(path):
                        csv_errors.append(f"{sym}: no server CSV @ tf {tf} (path: {path})")
                        continue
                    df = pd.read_csv(path)
                    df.columns = [c.strip().lower() for c in df.columns]
                    if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
                    for c in ("open","high","low","close"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
                    if len(df) < 30:
                        csv_errors.append(f"{sym}: not enough candles ({len(df)})")
                        continue
                    run_one(sym, df)
                except Exception as e:
                    csv_errors.append(f"{sym}: {e}")

        summary["winrate"] = round((summary["wins"]/summary["trades"])*100,2) if summary["trades"] else 0.0

        payload = {"summary": summary, "results": results, "tf": tf, "expiry": expiry, "strategy": strategy}
        if csv_errors:
            payload["warnings"] = csv_errors
            flash("Backtest completed with warnings. Check the table.")
        else:
            flash("Backtest complete.")
        session["bt"] = payload

    except Exception as e:
        session["bt"] = {"error": str(e), "warnings": csv_errors}
        flash(f"Backtest error: {e}")

    return redirect(url_for('dashboard.dashboard'))
