# Kishore FVG

A Streamlit backtesting app for a 5-minute Fair Value Gap (FVG) strategy. It can fetch Binance Vision Spot 5-minute candles for USDT pairs, upload exchange OHLC CSV files, or use Yahoo Finance where supported. The app does not place live trades.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Use the app

1. Open the sidebar from the top-left menu.
2. Select **Binance Vision Spot (USDT)**, choose a pair (BTCUSDT, ETHUSDT, XRPUSDT, or XAUTUSDT), and set UTC start/end dates. The app downloads daily 5-minute Spot kline archives directly from Binance Vision (maximum 61 days per run).
3. Set session timezone and strategy options, then click **Run backtest**.
4. If a pair or date has no Binance archive, the app reports unavailable dates. Availability depends on the Spot listing and Binance Vision archive. CSV upload remains available for other symbols/data.

Binance Vision archive format uses Unix-millisecond open timestamps followed by OHLC and volume fields. Data is converted to the configured session timezone before session rules are evaluated.

## Strategy rules implemented

- Detect the first 3-candle FVG during each configured Asia session.
- Wait for the configured delay (20 minutes by default), then enter opposite the FVG direction at the candle close.
- In the corresponding New York session, a candle close crossing either FVG boundary can trigger one trade. Choose **Opposite of cross** or **With cross**.
- Stop/target default to $17 loss and $25 profit per trade. The app converts those values to price distances using the configured quantity. Choose **Price move** for fixed price distances.
- One position at a time. Stop/target checks begin on candles after entry. If both are touched within one candle, the backtest assumes the stop was hit first. Trades still open at the end of data are marked at the final close.

Backtest results depend on session timezone, exchange data, quantity, fees, and slippage assumptions. Historical results do not guarantee future performance. This project is not financial advice.

## Deploy

Connect this repository to Streamlit Community Cloud and set the entry point to `app.py`.
