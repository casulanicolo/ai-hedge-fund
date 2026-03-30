"""
src/indicators/technical_indicators.py
Calcola tutti gli indicatori tecnici avanzati da un DataFrame OHLCV.
Nessuna chiamata API — tutto calcolato da dati già presenti nello stato.
"""

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(value, default=0.0) -> float:
    """Converte in float, restituisce default se NaN/None/invalido."""
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Indicatori base (già presenti nel vecchio technicals.py, mantenuti)
# ---------------------------------------------------------------------------

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = prices_df["close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20, num_std: float = 2.0):
    """Restituisce (upper, middle, lower)."""
    sma = prices_df["close"].rolling(window).mean()
    std = prices_df["close"].rolling(window).std()
    return sma + num_std * std, sma, sma - num_std * std


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    atr = calculate_atr(df, period)
    up   = df["high"] - df["high"].shift()
    down = df["low"].shift() - df["low"]
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(com=period-1, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(com=period-1, min_periods=period).mean()
    plus_di  = 100 * plus_dm_s  / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(com=period-1, min_periods=period).mean()
    return pd.DataFrame({"adx": adx, "+di": plus_di, "-di": minus_di})


# ---------------------------------------------------------------------------
# Indicatori nuovi
# ---------------------------------------------------------------------------

def calculate_macd(prices_df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    ema_fast = prices_df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = prices_df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    }


def calculate_vwap(prices_df: pd.DataFrame) -> pd.Series:
    """
    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    Calcolato su finestra rolling 20 giorni (adatto a dati daily).
    """
    typical = (prices_df["high"] + prices_df["low"] + prices_df["close"]) / 3
    window = min(20, len(prices_df))
    tp_vol = (typical * prices_df["volume"]).rolling(window).sum()
    vol    = prices_df["volume"].rolling(window).sum()
    return tp_vol / vol.replace(0, np.nan)


def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume: accumula volume con segno del movimento di prezzo.
    """
    direction = np.sign(prices_df["close"].diff().fillna(0))
    obv = (direction * prices_df["volume"]).cumsum()
    return obv


def calculate_ichimoku(prices_df: pd.DataFrame) -> dict:
    """
    Ichimoku Cloud:
    - Tenkan-sen  (conversion): (max9  + min9)  / 2
    - Kijun-sen   (base):       (max26 + min26) / 2
    - Senkou A:   (Tenkan + Kijun) / 2  [shifted forward 26]
    - Senkou B:   (max52 + min52)  / 2  [shifted forward 26]
    - Chikou:     close shifted back 26
    """
    high = prices_df["high"]
    low  = prices_df["low"]
    close = prices_df["close"]

    tenkan  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    kijun   = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou  = close.shift(-26)

    return {
        "tenkan":   tenkan,
        "kijun":    kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou":   chikou,
    }


def calculate_fibonacci_levels(prices_df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    Calcola i livelli di Fibonacci sul range high/low degli ultimi N giorni.
    Livelli standard: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    """
    recent = prices_df.tail(lookback)
    high = recent["high"].max()
    low  = recent["low"].min()
    diff = high - low

    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {f"fib_{int(r*100)}": round(high - diff * r, 4) for r in ratios}
    levels["swing_high"] = high
    levels["swing_low"]  = low
    return levels


def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """H < 0.5 mean-reverting, H ≈ 0.5 random walk, H > 0.5 trending."""
    lags = range(2, max_lag)
    tau  = [max(1e-8, np.sqrt(np.std(np.subtract(
        price_series.values[lag:], price_series.values[:-lag]
    )))) for lag in lags]
    try:
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(reg[0])
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Snapshot: dizionario scalari per l'ultimo giorno disponibile
# ---------------------------------------------------------------------------

def compute_indicator_snapshot(daily_df: pd.DataFrame, intraday_df: pd.DataFrame | None = None) -> dict:
    """
    Calcola tutti gli indicatori e restituisce un dizionario di scalari
    pronti per essere passati all'LLM o al regime detector.

    Args:
        daily_df:    DataFrame OHLCV daily (colonne: open, high, low, close, volume)
        intraday_df: DataFrame OHLCV intraday 5m (opzionale)

    Returns:
        dict con tutti i valori scalari dell'ultimo giorno.
    """
    if daily_df is None or len(daily_df) < 30:
        return {}

    df = daily_df.copy()

    # --- RSI ---
    rsi_14 = calculate_rsi(df, 14)
    rsi_28 = calculate_rsi(df, 28)

    # --- MACD ---
    macd   = calculate_macd(df)
    macd_val  = safe_float(macd["macd"].iloc[-1])
    macd_sig  = safe_float(macd["signal"].iloc[-1])
    macd_hist = safe_float(macd["histogram"].iloc[-1])

    # --- Bollinger Bands ---
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df)
    price_now = safe_float(df["close"].iloc[-1])
    bb_width  = safe_float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    bb_pct    = safe_float(
        (price_now - bb_lower.iloc[-1]) / bb_width if bb_width > 0 else 0.5
    )

    # --- ATR ---
    atr = calculate_atr(df, 14)
    atr_val = safe_float(atr.iloc[-1])
    atr_pct = safe_float(atr_val / price_now if price_now > 0 else 0)

    # --- ADX ---
    adx_df  = calculate_adx(df, 14)
    adx_val = safe_float(adx_df["adx"].iloc[-1])
    plus_di = safe_float(adx_df["+di"].iloc[-1])
    minus_di= safe_float(adx_df["-di"].iloc[-1])

    # --- VWAP ---
    vwap     = calculate_vwap(df)
    vwap_val = safe_float(vwap.iloc[-1])
    price_vs_vwap = safe_float((price_now - vwap_val) / vwap_val if vwap_val > 0 else 0)

    # --- OBV ---
    obv      = calculate_obv(df)
    obv_val  = safe_float(obv.iloc[-1])
    obv_ma   = safe_float(obv.rolling(20).mean().iloc[-1])
    obv_trend = "rising" if obv_val > obv_ma else "falling"

    # --- Ichimoku ---
    ichi     = calculate_ichimoku(df)
    tenkan   = safe_float(ichi["tenkan"].iloc[-1])
    kijun    = safe_float(ichi["kijun"].iloc[-1])
    senkou_a = safe_float(ichi["senkou_a"].iloc[-1])
    senkou_b = safe_float(ichi["senkou_b"].iloc[-1])
    cloud_top    = max(senkou_a, senkou_b)
    cloud_bottom = min(senkou_a, senkou_b)
    above_cloud  = price_now > cloud_top
    below_cloud  = price_now < cloud_bottom
    in_cloud     = not above_cloud and not below_cloud

    # --- Fibonacci ---
    fib = calculate_fibonacci_levels(df, lookback=60)
    # Nearest support: livelli fib sotto il prezzo corrente
    fib_supports = sorted(
        [v for k, v in fib.items() if k.startswith("fib_") and v < price_now],
        reverse=True
    )
    nearest_support = fib_supports[0] if fib_supports else safe_float(fib.get("swing_low", 0))

    # --- EMA trend ---
    ema_8  = safe_float(calculate_ema(df, 8).iloc[-1])
    ema_21 = safe_float(calculate_ema(df, 21).iloc[-1])
    ema_55 = safe_float(calculate_ema(df, 55).iloc[-1])

    # --- Hurst ---
    hurst = calculate_hurst_exponent(df["close"])

    # --- Intraday (5m) — solo se disponibile ---
    intraday_rsi  = None
    intraday_macd = None
    if intraday_df is not None and len(intraday_df) >= 30:
        intraday_rsi  = safe_float(calculate_rsi(intraday_df, 14).iloc[-1])
        intraday_macd_dict = calculate_macd(intraday_df)
        intraday_macd = safe_float(intraday_macd_dict["histogram"].iloc[-1])

    snapshot = {
        # Prezzo
        "price": price_now,

        # RSI
        "rsi_14": safe_float(rsi_14.iloc[-1]),
        "rsi_28": safe_float(rsi_28.iloc[-1]),

        # MACD
        "macd": macd_val,
        "macd_signal": macd_sig,
        "macd_histogram": macd_hist,
        "macd_bullish": macd_hist > 0,

        # Bollinger
        "bb_upper": safe_float(bb_upper.iloc[-1]),
        "bb_lower": safe_float(bb_lower.iloc[-1]),
        "bb_pct":   bb_pct,
        "bb_width": bb_width,

        # ATR
        "atr":     atr_val,
        "atr_pct": atr_pct,

        # ADX
        "adx":      adx_val,
        "plus_di":  plus_di,
        "minus_di": minus_di,
        "strong_trend": adx_val > 25,

        # VWAP
        "vwap":          vwap_val,
        "price_vs_vwap": price_vs_vwap,

        # OBV
        "obv":       obv_val,
        "obv_trend": obv_trend,

        # Ichimoku
        "tenkan":        tenkan,
        "kijun":         kijun,
        "senkou_a":      senkou_a,
        "senkou_b":      senkou_b,
        "above_cloud":   above_cloud,
        "below_cloud":   below_cloud,
        "in_cloud":      in_cloud,

        # Fibonacci
        "fib_levels":      fib,
        "nearest_support": nearest_support,

        # EMA
        "ema_8":  ema_8,
        "ema_21": ema_21,
        "ema_55": ema_55,

        # Hurst
        "hurst": hurst,

        # Intraday
        "intraday_rsi_14":      intraday_rsi,
        "intraday_macd_hist":   intraday_macd,
    }

    return snapshot
