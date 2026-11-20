import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Kishore Asia FVG", page_icon="📊", layout="wide")
st.title("Kishore Asia FVG — Signal & Backtest")
st.caption("5-minute Asia Session FVG scanner & backtest engine. Research only; does not place live orders.")

BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote", "ignore",
]


def in_session(ts, session_text):
    start_text, end_text = session_text.split("-")
    start = time.fromisoformat(start_text.strip())
    end = time.fromisoformat(end_text.strip())
    current = ts.timetz().replace(tzinfo=None)
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def trading_day_key(ts, asia_session):
    """Assign an overnight Asia session to the date on which it ends."""
    start_text, end_text = asia_session.split("-")
    start = time.fromisoformat(start_text.strip())
    end = time.fromisoformat(end_text.strip())
    current = ts.timetz().replace(tzinfo=None)
    if start > end and current >= start:
        return (ts + pd.Timedelta(days=1)).date()
    return ts.date()


def normalize_data(raw, session_timezone):
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"date": "timestamp", "datetime": "timestamp", "time": "timestamp", "vol": "volume"}
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    if "timestamp" not in df.columns:
        first = df.columns[0]
        if first.startswith("unnamed"):
            df = df.rename(columns={first: "timestamp"})
        else:
            raise ValueError("CSV needs a timestamp/date/datetime column.")
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing OHLC columns: " + ", ".join(missing))
    parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    valid = parsed.notna()
    df = df.loc[valid].copy()
    df.index = parsed.loc[valid].dt.tz_convert(session_timezone)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.empty:
        raise ValueError("No valid OHLC rows found.")
    return df


def fetch_binance_vision(symbol, start_date, end_date):
    """Fetch daily Binance Spot 5m kline archives for an inclusive date range (up to 365 days)."""
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if (end_date - start_date).days > 365:
        raise ValueError("Choose a date range of 365 days or less.")

    def fetch_day(day):
        date_text = day.isoformat()
        url = (
            "https://data.binance.vision/data/spot/daily/klines/"
            f"{symbol}/5m/{symbol}-5m-{date_text}.zip"
        )
        response = requests.get(
            url, timeout=25, headers={"User-Agent": "Kishore-FVG-Backtest/1.0"}
        )
        if response.status_code == 404:
            return day, None, "unavailable"
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV was found in the Binance archive for {date_text}.")
            frame = pd.read_csv(archive.open(csv_names[0]), header=None)
        if frame.empty:
            return day, None, "empty"

        first_val = str(frame.iloc[0, 0]).strip().lower()
        if first_val in ("open_time", "kline_open_time", "timestamp"):
            frame = frame.iloc[1:].reset_index(drop=True)

        frame = frame.iloc[:, :len(BINANCE_COLUMNS)]
        frame.columns = BINANCE_COLUMNS[:len(frame.columns)]

        raw_ts = pd.to_numeric(frame["open_time"], errors="coerce")
        valid_ts = raw_ts.dropna()
        if valid_ts.empty:
            return day, None, "empty"
        unit = "us" if valid_ts.median() > 1e14 else "ms"

        frame["timestamp"] = pd.to_datetime(raw_ts, unit=unit, utc=True)
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            return day, None, "empty"
        return day, frame[["timestamp", "open", "high", "low", "close", "volume"]], "ok"

    days = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
    frames, missing_days, failures = [], [], []
    progress = st.progress(0, text=f"Downloading {len(days)} days of 5-minute candles…")
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_day, day): day for day in days}
        for complete, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                _, frame, status = future.result()
                if status == "ok":
                    frames.append(frame)
                else:
                    missing_days.append(day.isoformat())
            except Exception as exc:
                failures.append(f"{day.isoformat()}: {exc}")
            if complete % max(1, len(days) // 25) == 0 or complete == len(days):
                progress.progress(complete / len(days), text=f"Downloaded {complete}/{len(days)} days")
    progress.empty()
    if not frames:
        detail = " | ".join(failures[:3]) if failures else "All requested days returned 404/empty."
        raise ValueError(f"No Binance Vision 5-minute data found for {symbol}. {detail}")
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    return data.drop_duplicates(subset="timestamp", keep="last"), missing_days, failures


def run_asia_fvg_strategy(df, asia_session, wait_minutes, investment_usd,
                          risk_mode, stop_value, target_value, fee_bps):
    rows = df[["open", "high", "low", "close"]].copy()
    trades, states, fvg_zones, position = [], {}, [], None

    def level_distances(qty):
        if risk_mode == "USD P&L per trade":
            return stop_value / qty, target_value / qty
        return stop_value, target_value

    for i, (ts, bar) in enumerate(rows.iterrows()):
        is_asia = in_session(ts, asia_session)
        day_key = trading_day_key(ts, asia_session)
        state = states.setdefault(day_key, {
            "fvg_top": None, "fvg_bottom": None, "fvg_bull": None,
            "fvg_time": None, "asia_traded": False,
        })
        exited_this_bar = False

        # Exit management (conservative SL first if both SL & TP touched on same bar)
        if position is not None and i > position["entry_i"]:
            qty = position["qty"]
            stop_dist, target_dist = level_distances(qty)
            if position["side"] == "Long":
                stop_price = position["entry_price"] - stop_dist
                target_price = position["entry_price"] + target_dist
                stop_hit, target_hit = bar["low"] <= stop_price, bar["high"] >= target_price
            else:
                stop_price = position["entry_price"] + stop_dist
                target_price = position["entry_price"] - target_dist
                stop_hit, target_hit = bar["high"] >= stop_price, bar["low"] <= target_price

            exit_price, reason = None, None
            if stop_hit:
                exit_price, reason = stop_price, "Stop loss"
            elif target_hit:
                exit_price, reason = target_price, "Target"

            if exit_price is not None:
                gross = (exit_price - position["entry_price"]) * position["direction"] * qty
                fees = (position["entry_price"] + exit_price) * qty * (fee_bps / 10000)
                trades.append({
                    "Day": str(day_key),
                    "Entry time": position["entry_time"],
                    "Exit time": ts,
                    "Side": position["side"],
                    "FVG Type": position["fvg_type"],
                    "Investment ($)": round(position["investment"], 2),
                    "Qty": round(qty, 6),
                    "Entry price": position["entry_price"],
                    "Exit price": exit_price,
                    "Gross P&L ($)": round(gross, 2),
                    "Fees ($)": round(fees, 2),
                    "Net P&L ($)": round(gross - fees, 2),
                    "Exit reason": reason,
                })
                position, exited_this_bar = None, True

        # Detect first valid FVG inside Asia session
        if i >= 2 and is_asia and state["fvg_top"] is None:
            prior_high, prior_low = rows.iloc[i - 2]["high"], rows.iloc[i - 2]["low"]
            previous_close = rows.iloc[i - 1]["close"]
            if bar["low"] > prior_high and previous_close > prior_high:
                state.update(
                    fvg_top=float(bar["low"]),
                    fvg_bottom=float(prior_high),
                    fvg_bull=True,
                    fvg_time=ts,
                )
                fvg_zones.append({
                    "day": str(day_key), "time": ts, "type": "Bullish FVG",
                    "top": float(bar["low"]), "bottom": float(prior_high),
                })
            elif bar["high"] < prior_low and previous_close < prior_low:
                state.update(
                    fvg_top=float(prior_low),
                    fvg_bottom=float(bar["high"]),
                    fvg_bull=False,
                    fvg_time=ts,
                )
                fvg_zones.append({
                    "day": str(day_key), "time": ts, "type": "Bearish FVG",
                    "top": float(prior_low), "bottom": float(bar["high"]),
                })

        # Enter opposite of Asia FVG after wait_minutes using fixed USD investment
        if position is None and not exited_this_bar:
            if (is_asia and state["fvg_time"] is not None and not state["asia_traded"]
                    and ts >= state["fvg_time"] + pd.Timedelta(minutes=wait_minutes)):
                entry_price = float(bar["close"])
                qty = investment_usd / entry_price
                direction = -1 if state["fvg_bull"] else 1
                side = "Long" if direction == 1 else "Short"
                fvg_label = "Bullish FVG -> Short" if state["fvg_bull"] else "Bearish FVG -> Long"
                position = {
                    "entry_i": i,
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "investment": investment_usd,
                    "qty": qty,
                    "direction": direction,
                    "side": side,
                    "fvg_type": fvg_label,
                }
                state["asia_traded"] = True

    # Build latest signal summary from the most recent trading day
    latest_ts = rows.index[-1]
    latest_close = float(rows.iloc[-1]["close"])
    latest_day = trading_day_key(latest_ts, asia_session)
    latest_state = states.get(latest_day, {})
    preview_qty = investment_usd / latest_close if latest_close > 0 else 1.0
    stop_dist, target_dist = level_distances(preview_qty)

    signal_info = {
        "latest_ts": latest_ts,
        "latest_close": latest_close,
        "in_asia_now": in_session(latest_ts, asia_session),
        "fvg_found": latest_state.get("fvg_top") is not None,
        "fvg_bull": latest_state.get("fvg_bull"),
        "fvg_top": latest_state.get("fvg_top"),
        "fvg_bottom": latest_state.get("fvg_bottom"),
        "fvg_time": latest_state.get("fvg_time"),
        "traded": latest_state.get("asia_traded", False),
        "open_position": position,
        "stop_dist": stop_dist,
        "target_dist": target_dist,
        "investment_usd": investment_usd,
    }

    if position is not None:
        ts, exit_price = rows.index[-1], float(rows.iloc[-1]["close"])
        qty = position["qty"]
        gross = (exit_price - position["entry_price"]) * position["direction"] * qty
        fees = (position["entry_price"] + exit_price) * qty * (fee_bps / 10000)
        trades.append({
            "Day": str(trading_day_key(ts, asia_session)),
            "Entry time": position["entry_time"],
            "Exit time": ts,
            "Side": position["side"],
            "FVG Type": position["fvg_type"],
            "Investment ($)": round(position["investment"], 2),
            "Qty": round(qty, 6),
            "Entry price": position["entry_price"],
            "Exit price": exit_price,
            "Gross P&L ($)": round(gross, 2),
            "Fees ($)": round(fees, 2),
            "Net P&L ($)": round(gross - fees, 2),
            "Exit reason": "End of data (Open)",
        })

    return pd.DataFrame(trades), pd.DataFrame(fvg_zones), signal_info


def build_chart(df, trades, fvg_zones, title):
    # If dataset is very large (> 60 days), show last 30 days on interactive chart for fast rendering
    chart_df = df.last("30D") if len(df) > 18000 else df
    cutoff = chart_df.index.min()

    fig = go.Figure(data=[go.Candlestick(
        x=chart_df.index, open=chart_df["open"], high=chart_df["high"],
        low=chart_df["low"], close=chart_df["close"], name="Price"
    )])

    if not fvg_zones.empty:
        recent_zones = fvg_zones[fvg_zones["time"] >= cutoff].tail(40)
        for _, z in recent_zones.iterrows():
            color = "rgba(0, 200, 83, 0.20)" if z["type"] == "Bullish FVG" else "rgba(255, 82, 82, 0.20)"
            fig.add_shape(
                type="rect",
                x0=z["time"],
                x1=z["time"] + pd.Timedelta(hours=4),
                y0=z["bottom"],
                y1=z["top"],
                fillcolor=color,
                line=dict(width=0),
            )

    if not trades.empty:
        chart_trades = trades[trades["Entry time"] >= cutoff]
        longs = chart_trades[chart_trades["Side"] == "Long"]
        shorts = chart_trades[chart_trades["Side"] == "Short"]
        if not longs.empty:
            fig.add_trace(go.Scatter(
                x=longs["Entry time"], y=longs["Entry price"], mode="markers",
                marker={"color": "#00e676", "size": 11, "symbol": "triangle-up"},
                name="Long Entry", text=longs["FVG Type"],
                hovertemplate="%{text}<br>%{x}<br>Price: %{y}<extra></extra>"
            ))
        if not shorts.empty:
            fig.add_trace(go.Scatter(
                x=shorts["Entry time"], y=shorts["Entry price"], mode="markers",
                marker={"color": "#ff1744", "size": 11, "symbol": "triangle-down"},
                name="Short Entry", text=shorts["FVG Type"],
                hovertemplate="%{text}<br>%{x}<br>Price: %{y}<extra></extra>"
            ))
        exits = chart_trades[chart_trades["Exit time"] >= cutoff]
        if not exits.empty:
            fig.add_trace(go.Scatter(
                x=exits["Exit time"], y=exits["Exit price"], mode="markers",
                marker={"color": "#ffd600", "size": 8, "symbol": "x"},
                name="Exits", text=exits["Exit reason"],
                hovertemplate="Exit: %{text}<br>%{x}<br>Price: %{y}<extra></extra>"
            ))

    fig.update_layout(
        title=title + (" (Chart showing last 30 days)" if len(df) > 18000 else ""),
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=650,
    )
    return fig


with st.sidebar:
    st.header("Asia FVG Settings")
    market = st.selectbox("Market", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "Custom symbol"])
    source = st.radio("Data source", ["Binance Vision Spot (USDT)", "Upload CSV", "Yahoo Finance (5-minute)"])
    session_timezone = st.text_input("Session timezone", "America/New_York")
    asia_session = st.text_input("Asia session (session timezone)", "1900-0400")

    if source == "Binance Vision Spot (USDT)":
        binance_symbol = (
            st.text_input("Binance Spot symbol", "BTCUSDT").strip().upper()
            if market == "Custom symbol" else market
        )
        today_utc = datetime.now(timezone.utc).date()
        period_preset = st.selectbox(
            "Backtest period",
            ["Last 14 Days", "Last 30 Days", "Last 90 Days (3 Months)", "Last 180 Days (6 Months)", "Last 365 Days (1 Year)", "Custom Date Range"],
            index=1,
        )
        preset_days = {
            "Last 14 Days": 14,
            "Last 30 Days": 30,
            "Last 90 Days (3 Months)": 90,
            "Last 180 Days (6 Months)": 180,
            "Last 365 Days (1 Year)": 365,
        }
        if period_preset == "Custom Date Range":
            date_end = st.date_input("End date (UTC)", value=today_utc - timedelta(days=1))
            date_start = st.date_input("Start date (UTC)", value=today_utc - timedelta(days=30))
        else:
            date_end = today_utc - timedelta(days=1)
            date_start = date_end - timedelta(days=preset_days[period_preset] - 1)
            st.caption(f"Range: {date_start} to {date_end} ({preset_days[period_preset]} days)")
    elif source == "Yahoo Finance (5-minute)":
        yahoo_ticker = st.text_input("Yahoo ticker", market if market != "Custom symbol" else "BTC-USD")
    else:
        uploaded = st.file_uploader("Upload 5-minute OHLC CSV", type=["csv"])

    wait_minutes = st.number_input("Wait after Asia FVG (minutes)", min_value=1, value=20, step=1)
    investment_usd = st.number_input("Investment per trade ($)", min_value=10.0, value=1000.0, step=100.0)
    risk_mode = st.selectbox("Stop/target interpretation", ["USD P&L per trade", "Price move"])
    stop_value = st.number_input("Stop loss ($ P&L or price move)", min_value=0.01, value=17.0, step=1.0)
    target_value = st.number_input("Target ($ P&L or price move)", min_value=0.01, value=25.0, step=1.0)
    fee_bps = st.number_input("Fee per side (basis points)", min_value=0.0, value=5.0, step=1.0)
    run = st.button("Run Signal & Backtest", type="primary", use_container_width=True)

if run:
    try:
        if source == "Binance Vision Spot (USDT)":
            raw, missing_days, download_failures = fetch_binance_vision(binance_symbol, date_start, date_end)
            df = normalize_data(raw, session_timezone)
            title = f"Binance Spot {binance_symbol} · 5m (Asia FVG)"
            if missing_days:
                st.warning(f"No archive available for {len(missing_days)} day(s); loaded available days.")
            if download_failures:
                st.warning(f"Failed to download {len(download_failures)} day(s).")
        elif source == "Upload CSV":
            if uploaded is None:
                st.error("Upload an OHLC CSV first.")
                st.stop()
            raw = pd.read_csv(uploaded)
            df = normalize_data(raw, session_timezone)
            title = "Uploaded Data · 5m (Asia FVG)"
        else:
            import yfinance as yf
            downloaded = yf.download(yahoo_ticker, period="60d", interval="5m", progress=False, auto_adjust=False)
            if downloaded.empty:
                raise ValueError("Yahoo Finance returned no 5-minute data. Try Binance Vision or CSV.")
            if isinstance(downloaded.columns, pd.MultiIndex):
                downloaded.columns = downloaded.columns.get_level_values(0)
            downloaded = downloaded.reset_index()
            downloaded.columns = [str(c).strip().lower() for c in downloaded.columns]
            if "datetime" in downloaded.columns:
                downloaded = downloaded.rename(columns={"datetime": "timestamp"})
            elif "date" in downloaded.columns:
                downloaded = downloaded.rename(columns={"date": "timestamp"})
            df = normalize_data(downloaded, session_timezone)
            title = f"{yahoo_ticker} · 5m (Asia FVG)"

        if len(df) < 3:
            raise ValueError("At least 3 OHLC bars are needed to detect an FVG.")

        trades, fvg_zones, sig = run_asia_fvg_strategy(
            df, asia_session, int(wait_minutes), float(investment_usd),
            risk_mode, float(stop_value), float(target_value), float(fee_bps)
        )

        # --- SECTION 1: LATEST SIGNAL STATUS ---
        st.subheader("📡 Latest Session Signal Status")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Latest Candle Close", f"${sig['latest_close']:,.2f}", str(sig["latest_ts"]))

        if sig["open_position"] is not None:
            pos = sig["open_position"]
            side = pos["side"]
            entry = pos["entry_price"]
            pos_stop, pos_target = (
                (stop_value / pos["qty"], target_value / pos["qty"])
                if risk_mode == "USD P&L per trade" else (stop_value, target_value)
            )
            sl = entry - pos_stop if side == "Long" else entry + pos_stop
            tp = entry + pos_target if side == "Long" else entry - pos_target
            s2.metric("Active Signal", f"OPEN {side.upper()} (${pos['investment']:,.0f})", f"Entry @ {entry:,.2f}")
            s3.metric("Stop Loss", f"{sl:,.2f}")
            s4.metric("Take Profit", f"{tp:,.2f}")
        elif sig["fvg_found"]:
            fvg_kind = "Bullish FVG" if sig["fvg_bull"] else "Bearish FVG"
            next_side = "SHORT" if sig["fvg_bull"] else "LONG"
            trigger_time = sig["fvg_time"] + pd.Timedelta(minutes=int(wait_minutes))
            status_label = "EXECUTED" if sig["traded"] else f"PENDING ({next_side})"
            s2.metric("Asia FVG Detected", fvg_kind, f"{sig['fvg_bottom']:,.2f} - {sig['fvg_top']:,.2f}")
            s3.metric("Signal Action", status_label, f"Trigger >= {trigger_time.strftime('%H:%M')}")
            s4.metric("Per Trade Size", f"${sig['investment_usd']:,.0f}", "In Asia Session" if sig["in_asia_now"] else "Outside Asia")
        else:
            s2.metric("Asia FVG Detected", "None Yet")
            s3.metric("Signal Action", "WAITING FOR FVG")
            s4.metric("Per Trade Size", f"${sig['investment_usd']:,.0f}", "In Asia Session" if sig["in_asia_now"] else "Outside Asia")

        st.divider()

        # --- SECTION 2: BACKTEST RESULTS ---
        st.subheader(f"📊 Backtest Results — {title}")
        st.caption(f"Loaded {len(df):,} five-minute candles from {df.index.min()} to {df.index.max()} | Investment per trade: ${investment_usd:,.2f}.")

        if trades.empty:
            st.warning("No completed or triggered Asia FVG trades found in this date range.")
        else:
            net = trades["Net P&L ($)"].sum()
            wins = (trades["Net P&L ($)"] > 0).sum()
            loss_sum = trades.loc[trades["Net P&L ($)"] < 0, "Net P&L ($)"].sum()
            profit_factor = (
                trades.loc[trades["Net P&L ($)"] > 0, "Net P&L ($)"].sum() / abs(loss_sum)
                if loss_sum != 0 else None
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Net P&L", f"${net:,.2f}")
            m2.metric("Trades", f"{len(trades)}")
            m3.metric("Win Rate", f"{wins / len(trades) * 100:.1f}%")
            m4.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor is not None else "—")

            st.plotly_chart(build_chart(df, trades, fvg_zones, title), use_container_width=True)

            st.subheader("Trade History")
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Trades CSV",
                trades.to_csv(index=False).encode("utf-8"),
                file_name="kishore_asia_fvg_trades.csv",
                mime="text/csv",
            )

        st.caption("Assumptions: Entries at candle close after wait period; SL/TP checked on subsequent bars (SL assumed first if both touched in same bar).")
    except Exception as exc:
        st.error(f"Could not run backtest: {exc}")
else:
    st.info("Open the sidebar (top-left) to choose your pair, backtest period, and per-trade investment ($1,000 default), then click **Run Signal & Backtest**.")
