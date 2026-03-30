import yfinance as yf
import pandas as pd
from src.indicators.technical_indicators import compute_indicator_snapshot
from src.indicators.regime_detector import detect_regime

# Scarica dati AAPL reali
df = yf.download('AAPL', period='6mo', interval='1d', auto_adjust=True, progress=False)
# yfinance restituisce MultiIndex se auto_adjust=True — lo appiattisce
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0].lower() for col in df.columns]
else:
    df.columns = [c.lower() for c in df.columns]

print('Colonne DataFrame:', list(df.columns))
print('Righe:', len(df))
df = df.reset_index(drop=True)

snapshot = compute_indicator_snapshot(df, None)
regime = detect_regime(snapshot)

print('=== SNAPSHOT (valori chiave) ===')
for k in ['price', 'rsi_14', 'macd_histogram', 'bb_pct', 'adx', 'obv_trend', 'above_cloud', 'hurst']:
    print(f'  {k}: {snapshot.get(k)}')

print()
print('=== REGIME ===')
print(f'  Regime:     {regime["regime"]}')
print(f'  Direction:  {regime["direction"]}')
print(f'  Confidence: {regime["confidence"]:.0%}')
for r in regime['reasons']:
    print(f'  - {r}')
