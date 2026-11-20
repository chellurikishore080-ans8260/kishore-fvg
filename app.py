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
    """Fetch daily Binance Spot 5m kline archives for an inclusive date range."""
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if (end_date - start_date).days > 60:
        raise ValueError("Choose a date range of 61 days or less to keep downloads fast.")

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
        # Fix for Binance microseconds vs milliseconds timestamps
        unit = "us" if valid_ts.median() > 1e14 else "ms"

        frame["timestamp"] = pd.to_datetime(raw_ts, unit=unit, utc=True)
        frame = frame.dropna(subset=["timestamp"])
        if frame.empty:
            return day, None, "empty"
        return day, frame[["timestamp", "open", "high", "low", "close", "volume"]], "ok"

    days = [start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)]
    frames, missing_days, failures = [], [], []
    progress = st.progress(0, text="Downloading Binance Vision 5-minute candles…")
    with ThreadPoolExecutor(max_workers=6) as pool:
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
            progress.progress(complete / len(days), text=f"Downloaded {complete}/{len(days)} days")
    progress.empty()
    if not frames:
        detail = " | ".join(failures[:3]) if failures else "All requested days returned 404/empty."
        raise ValueError(f"No Binance Vision 5-minute data found for {symbol}. {detail}")
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    return data.drop_duplicates(subset="timestamp", keep="last"), missing_days, failures


def run_asia_fvg_strategy(df, asia_session, wait_minutes, qty,
                          risk_mode, stop_value, target_value, fee_bps):
    rows = df[["open", "high", "low", "close"]].copy()
    trades, states, fvg_zones, position = [], {}, [], None

    def level_distances():
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
            stop_dist, target_dist = level_distances()
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
                    "Entry price": position["entry_price"],
                    "Exit price": exit_price,
                    "Gross P&L ($)": gross,
                    "Fees ($)": fees,
                    "Net P&L ($)": gross - fees,
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

        # Enter opposite of Asia FVG after wait_minutes
        if position is None and not exited_this_bar:
            if (is_asia and state["fvg_time"] is not None and not state["asia_traded"]
                    and ts >= state["fvg_time"] + pd.Timedelta(minutes=wait_minutes)):
                direction = -1 if state["fvg_bull"] else 1
                side = "Long" if direction == 1 else "Short"
                fvg_label = "Bullish FVG -> Short" if state["fvg_bull"] else "Bearish FVG -> Long"
                position = {
                    "entry_i": i,
                    "entry_time": ts,
                    "entry_price": float(bar["close"]),
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
    stop_dist, target_dist = level_distances()

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
    }

    if position is not None:
        ts, exit_price = rows.index[-1], float(rows.iloc[-1]["close"])
        gross = (exit_price - position["entry_price"]) * position["direction"] * qty
        fees = (position["entry_price"] + exit_price) * qty * (fee_bps / 10000)
        trades.append({
            "Day": str(trading_day_key(ts, asia_session)),
            "Entry time": position["entry_time"],
            "Exit time": ts,
            "Side": position["side"],
            "FVG Type": position["fvg_type"],
            "Entry price": position["entry_price"],
            "Exit price": exit_price,
            "Gross P&L ($)": gross,
            "Fees ($)": fees,
            "Net P&L ($)": gross - fees,
            "Exit reason": "End of data (Open)",
        })

    return pd.DataFrame(trades), pd.DataFrame(fvg_zones), signal_info


def build_chart(df, trades, fvg_zones, title):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
    )])

    # Plot recent FVG zones (last 30 zones so chart remains fast and clean)
    if not fvg_zones.empty:
        for _, z in fvg_zones.tail(30).iterrows():
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
        longs = trades[trades["Side"] == "Long"]
        shorts = trades[trades["Side"] == "Short"]
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
        fig.add_trace(go.Scatter(
            x=trades["Exit time"], y=trades["Exit price"], mode="markers",
            marker={"color": "#ffd600", "size": 8, "symbol": "x"},
            name="Exits", text=trades["Exit reason"],
            hovertemplate="Exit: %{text}<br>%{x}<br>Price: %{y}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
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
        date_end = st.date_input("End date (UTC)", value=today_utc - timedelta(days=1))
        date_start = st.date_input("Start date (UTC)", value=today_utc - timedelta(days=14))
        st.caption("Daily 5-minute Spot archives; max range 61 days.")
    elif source == "Yahoo Finance (5-minute)":
        yahoo_ticker = st.text_input("Yahoo ticker", market if market != "Custom symbol" else "BTC-USD")
    else:
        uploaded = st.file_uploader("Upload 5-minute OHLC CSV", type=["csv"])

    wait_minutes = st.number_input("Wait after Asia FVG (minutes)", min_value=1, value=20, step=1)
    quantity = st.number_input("Position quantity (coin/unit)", min_value=0.000001, value=1.0, format="%.6f")
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
            df, asia_session, int(wait_minutes), float(quantity),
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
            sl = entry - sig["stop_dist"] if side == "Long" else entry + sig["stop_dist"]
            tp = entry + sig["target_dist"] if side == "Long" else entry - sig["target_dist"]
            s2.metric("Active Signal", f"OPEN {side.upper()}", f"Entry @ {entry:,.2f}")
            s3.metric("Stop Loss", f"{sl:,.2f}")
            s4.metric("Take Profit", f"{tp:,.2f}")
        elif sig["fvg_found"]:
            fvg_kind = "Bullish FVG" if sig["fvg_bull"] else "Bearish FVG"
            next_side = "SHORT" if sig["fvg_bull"] else "LONG"
            trigger_time = sig["fvg_time"] + pd.Timedelta(minutes=int(wait_minutes))
            status_label = "EXECUTED" if sig["traded"] else f"PENDING ({next_side})"
            s2.metric("Asia FVG Detected", fvg_kind, f"{sig['fvg_bottom']:,.2f} - {sig['fvg_top']:,.2f}")
            s3.metric("Signal Action", status_label, f"Trigger >= {trigger_time.strftime('%H:%M')}")
            s4.metric("Session State", "In Asia Session" if sig["in_asia_now"] else "Outside Asia")
        else:
            s2.metric("Asia FVG Detected", "None Yet")
            s3.metric("Signal Action", "WAITING FOR FVG")
            s4.metric("Session State", "In Asia Session" if sig["in_asia_now"] else "Outside Asia")

        st.divider()

        # --- SECTION 2: BACKTEST RESULTS ---
        st.subheader(f"📊 Backtest Results — {title}")
        st.caption(f"Loaded {len(df):,} five-minute candles from {df.index.min()} to {df.index.max()}.")

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
    st.info("Open the sidebar (top-left) to choose your pair, dates, Asia session window, and risk settings, then click **Run Signal & Backtest**.")
