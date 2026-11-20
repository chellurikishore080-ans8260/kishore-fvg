# Kishore FVG

A Streamlit backtesting app for a 5-minute Fair Value Gap (FVG) strategy. It can analyze BTC, ETH, XRP, and XAUT price data, including Asia-session FVGs and New York-session FVG crosses. It does not place live trades.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. To publish this app, connect this repository to Streamlit Community Cloud and set `app.py` as the entry point.

## Data

- **CSV upload:** Provide 5-minute OHLC candles with columns `timestamp, open, high, low, close`. `volume` is optional. Choose the source timezone for naive timestamps; timezone-aware timestamps are converted automatically.
- **Yahoo Finance:** The app requests the most recent 60 days of 5-minute data. Availability and ticker support vary; XAUT may require a data-provider CSV.
- For BTC, ETH, XRP, and XAUT, run a separate backtest using the actual exchange symbol/data you trade. Crypto exchange symbols are not necessarily the same as Yahoo tickers.

## Strategy rules implemented

1. Detect the first 3-candle FVG during each configured Asia session.
2. After the configured wait (20 minutes by default), enter opposite the FVG direction at that candle's close.
3. In the corresponding New York session, a close crossing either FVG boundary can trigger one trade. Choose **Opposite of cross** or **With cross** in the settings.
4. Stop and target default to a $17 loss and $25 profit **per trade**. The app converts those values to price distances using the configured quantity. Select **Price move** if you instead want fixed price distances.
5. One position at a time. Stop/target are evaluated on candles after entry. If both levels fall within one candle, the backtest assumes the stop was hit first. Trades still open at the end of the data are marked to the final close.

The backtest enters at bar close, estimates fees, and is not a guarantee of future performance. Verify session timezone, exchange data, quantity, fees, and slippage before interpreting results. This project is not financial advice.
