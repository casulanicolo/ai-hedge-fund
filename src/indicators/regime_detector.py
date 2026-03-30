"""
src/indicators/regime_detector.py
Classifica il regime di mercato corrente in TRENDING, RANGING o VOLATILE
usando i valori dello snapshot degli indicatori tecnici.
"""


def detect_regime(snapshot: dict) -> dict:
    """
    Determina il regime di mercato basandosi sugli indicatori calcolati.

    Logica:
    - VOLATILE:  ATR% alto (> 3%) OPPURE BB molto largo (> 8% del prezzo)
    - TRENDING:  ADX > 25 E prezzo fuori dalla nuvola Ichimoku
    - RANGING:   tutto il resto (mercato laterale)

    Args:
        snapshot: dizionario restituito da compute_indicator_snapshot()

    Returns:
        dict con:
          - regime:     "TRENDING" | "RANGING" | "VOLATILE"
          - direction:  "UP" | "DOWN" | "FLAT"  (rilevante solo in TRENDING)
          - confidence: float 0-1
          - reasons:    list[str] — spiegazioni leggibili dall'LLM
    """
    if not snapshot:
        return {
            "regime": "RANGING",
            "direction": "FLAT",
            "confidence": 0.3,
            "reasons": ["Dati insufficienti per la classificazione del regime."],
        }

    reasons = []

    # --- Raccogliamo i valori chiave ---
    adx       = snapshot.get("adx", 20.0)
    atr_pct   = snapshot.get("atr_pct", 0.01)
    bb_width  = snapshot.get("bb_width", 0.0)
    price     = snapshot.get("price", 1.0)
    bb_width_pct = bb_width / price if price > 0 else 0.0

    above_cloud  = snapshot.get("above_cloud", False)
    below_cloud  = snapshot.get("below_cloud", False)
    strong_trend = snapshot.get("strong_trend", False)   # ADX > 25

    plus_di  = snapshot.get("plus_di", 20.0)
    minus_di = snapshot.get("minus_di", 20.0)
    hurst    = snapshot.get("hurst", 0.5)

    ema_8   = snapshot.get("ema_8",  price)
    ema_21  = snapshot.get("ema_21", price)
    ema_55  = snapshot.get("ema_55", price)

    obv_trend = snapshot.get("obv_trend", "flat")

    # -----------------------------------------------------------------------
    # 1. VOLATILE — priorità massima
    #    Volatilità anomala: ATR% > 3% oppure BB width > 8% del prezzo
    # -----------------------------------------------------------------------
    volatile_score = 0.0

    if atr_pct > 0.03:
        volatile_score += 0.5
        reasons.append(f"ATR% elevato ({atr_pct:.1%}) — alta volatilità intraday.")

    if bb_width_pct > 0.08:
        volatile_score += 0.4
        reasons.append(f"Bollinger Bands molto larghe ({bb_width_pct:.1%} del prezzo).")

    if hurst > 0.65:
        volatile_score += 0.1
        reasons.append(f"Hurst ({hurst:.2f}) suggerisce momentum elevato.")

    if volatile_score >= 0.5:
        direction = _direction(plus_di, minus_di, ema_8, ema_21)
        return {
            "regime":     "VOLATILE",
            "direction":  direction,
            "confidence": min(volatile_score, 1.0),
            "reasons":    reasons,
        }

    # -----------------------------------------------------------------------
    # 2. TRENDING — ADX forte + prezzo fuori dalla nuvola Ichimoku
    # -----------------------------------------------------------------------
    trending_score = 0.0

    if strong_trend:
        trending_score += 0.4
        reasons.append(f"ADX {adx:.1f} > 25 — trend strutturato presente.")

    if above_cloud:
        trending_score += 0.35
        reasons.append("Prezzo sopra la nuvola Ichimoku — trend rialzista.")
    elif below_cloud:
        trending_score += 0.35
        reasons.append("Prezzo sotto la nuvola Ichimoku — trend ribassista.")

    if ema_8 > ema_21 > ema_55:
        trending_score += 0.15
        reasons.append("EMA 8 > 21 > 55 — allineamento rialzista.")
    elif ema_8 < ema_21 < ema_55:
        trending_score += 0.15
        reasons.append("EMA 8 < 21 < 55 — allineamento ribassista.")

    if obv_trend == "rising" and above_cloud:
        trending_score += 0.1
        reasons.append("OBV in crescita conferma il trend rialzista.")
    elif obv_trend == "falling" and below_cloud:
        trending_score += 0.1
        reasons.append("OBV in calo conferma il trend ribassista.")

    if trending_score >= 0.5:
        direction = _direction(plus_di, minus_di, ema_8, ema_21)
        return {
            "regime":     "TRENDING",
            "direction":  direction,
            "confidence": min(trending_score, 1.0),
            "reasons":    reasons,
        }

    # -----------------------------------------------------------------------
    # 3. RANGING — mercato laterale (default)
    # -----------------------------------------------------------------------
    ranging_score = 1.0 - max(volatile_score, trending_score)
    in_cloud = snapshot.get("in_cloud", False)

    if in_cloud:
        reasons.append("Prezzo dentro la nuvola Ichimoku — indecisione.")
    if hurst < 0.45:
        reasons.append(f"Hurst ({hurst:.2f}) < 0.5 — serie mean-reverting.")
    if adx < 20:
        reasons.append(f"ADX {adx:.1f} < 20 — trend debole o assente.")

    if not reasons:
        reasons.append("Nessun segnale di trend o volatilità dominante — mercato laterale.")

    return {
        "regime":     "RANGING",
        "direction":  "FLAT",
        "confidence": min(max(ranging_score, 0.3), 0.9),
        "reasons":    reasons,
    }


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _direction(plus_di: float, minus_di: float, ema_8: float, ema_21: float) -> str:
    """Determina la direzione del movimento prevalente."""
    score = 0
    if plus_di > minus_di:
        score += 1
    else:
        score -= 1
    if ema_8 > ema_21:
        score += 1
    else:
        score -= 1

    if score > 0:
        return "UP"
    elif score < 0:
        return "DOWN"
    return "FLAT"
