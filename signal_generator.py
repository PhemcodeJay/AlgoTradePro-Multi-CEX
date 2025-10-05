import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from indicators import scan_multiple_symbols, get_top_symbols
from ml import MLFilter
from notifications import send_all_notifications
from db import Signal, db_manager
from logging_config import get_logger
import os

logger = get_logger(__name__)

EXCHANGE = os.getenv("EXCHANGE", "binance").lower()
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

# Core Signal Utilities

def get_symbols(limit: int = 50) -> List[str]:
    """Fetch USDT-based trading pairs without using exchange clients."""
    # Predefined list of common USDT pairs supported by both Binance and Bybit
    usdt_symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT",
        "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT"
    ]

    # Adjust list based on exchange if needed
    if EXCHANGE == "bybit":
        # Bybit-specific futures pairs (optional, add if needed)
        usdt_symbols.extend(["NEARUSDT", "ALGOUSDT", "TRXUSDT"])
    elif EXCHANGE == "binance":
        # Binance-specific pairs (optional, add if needed)
        usdt_symbols.extend(["TRXUSDT", "VETUSDT", "ALGOUSDT"])

    # Remove duplicates and ensure uniqueness
    usdt_symbols = list(dict.fromkeys(usdt_symbols))

    # Sort by volume using get_top_symbols
    try:
        top_symbols = get_top_symbols(limit)
        usdt_symbols = [s for s in top_symbols if s in usdt_symbols][:limit]
    except Exception as e:
        logger.warning(f"Error sorting symbols by volume: {e}")
        # If get_top_symbols fails, return unsorted list truncated to limit
        usdt_symbols = usdt_symbols[:limit]

    if not usdt_symbols:
        logger.warning("No USDT symbols available, using fallback list")
        usdt_symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

    logger.info(f"Retrieved {len(usdt_symbols)} USDT trading pairs for {EXCHANGE}")
    return sorted(usdt_symbols)

def calculate_signal_score(analysis: Dict[str, Any]) -> float:
    """Calculate a score for a trading signal based on indicators"""
    score = analysis.get("score", 0)
    indicators = analysis.get("indicators", {})

    rsi = indicators.get("rsi", 50)
    if 20 <= rsi <= 30 or 70 <= rsi <= 80:
        score += 10
    elif rsi < 20 or rsi > 80:
        score += 5

    macd_hist = indicators.get("macd_histogram", 0)
    if abs(macd_hist) > 0.01:
        score += 8

    vol_ratio = indicators.get("volume_ratio", 1)
    if vol_ratio > 2:
        score += 12
    elif vol_ratio > 1.5:
        score += 6

    vol = indicators.get("volatility", 0)
    if 0.5 <= vol <= 3:
        score += 5
    elif vol > 5:
        score -= 10

    trend_score = indicators.get("trend_score", 0)
    score += trend_score * 3

    return min(100, max(0, score))

def enhance_signal(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance signal with additional calculated fields"""
    indicators = analysis.get("indicators", {})
    price = indicators.get("price", 0)
    atr = indicators.get("atr", 0)
    side = analysis.get("side", "BUY")
    leverage = 10
    atr_multiplier = 2
    risk_reward = 2

    if side.lower() == "buy":
        sl = price - atr * atr_multiplier
        tp = price + atr * atr_multiplier * risk_reward
        liquidation = price * (1 - 0.9 / leverage)
        trail = atr * 1.5
    else:
        sl = price + atr * atr_multiplier
        tp = price - atr * atr_multiplier * risk_reward
        liquidation = price * (1 + 0.9 / leverage)
        trail = -atr * 1.5

    margin_usdt = price * 1 / leverage  # Dummy qty=1

    enhanced = analysis.copy()
    enhanced.update({
        'entry': price,
        'sl': sl,
        'tp': tp,
        'trail': trail,
        'liquidation': liquidation,
        'leverage': leverage,
        'margin_usdt': margin_usdt,
        'market': 'futures',
        'created_at': datetime.now(timezone.utc)
    })
    return enhanced

def generate_signals(symbols: List[str], interval: str = "60", top_n: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
    """Generate and filter signals"""
    raw_analyses = scan_multiple_symbols(symbols, interval)
    
    scored = []
    for analysis in raw_analyses:
        analysis['score'] = calculate_signal_score(analysis)
        scored.append(analysis)
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    top = scored[:top_n]
    
    enhanced = [enhance_signal(a) for a in top]
    
    ml_filter = MLFilter()
    filtered = ml_filter.filter_signals(enhanced, threshold=threshold)
    
    # Save to DB with exchange
    for f in filtered:
        signal_obj = Signal(
            symbol=str(f.get("symbol") or "UNKNOWN"),
            interval=interval,
            signal_type=str(f.get("signal_type", "BUY")),
            side=str(f.get("side", "BUY")),
            score=float(f.get("score", 0)),
            entry=float(f.get("entry") or 0),
            sl=float(f.get("sl") or 0),
            tp=float(f.get("tp") or 0),
            trail=float(f.get("trail") or 0),
            liquidation=float(f.get("liquidation") or 0),
            leverage=int(f.get("leverage", 10)),
            margin_usdt=float(f.get("margin_usdt") or 0),
            market=str(f.get("market", "Unknown")),
            indicators=f.get("indicators", {}),
            exchange=EXCHANGE,
            created_at=f.get("created_at") or datetime.now(timezone.utc)
        )
        db_manager.add_signal(signal_obj)
    
    return filtered

# Signal Summary

def get_signal_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize signal statistics"""
    if not signals:
        return {"total": 0, "avg_score": 0, "top_symbol": "None"}

    total_signals = len(signals)
    avg_score = sum(float(s.get("score", 0)) for s in signals) / total_signals
    top_signal = max(signals, key=lambda x: float(x.get("score", 0)))
    top_symbol = top_signal.get("symbol", "Unknown")

    buy_signals = sum(1 for s in signals if s.get("side", "").upper() in ["BUY", "LONG"])
    sell_signals = total_signals - buy_signals

    market_types = {}
    for s in signals:
        market_types[s.get("market", "Unknown")] = market_types.get(s.get("market", "Unknown"), 0) + 1

    return {
        "total": total_signals,
        "avg_score": round(avg_score, 1),
        "top_symbol": top_symbol,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "market_distribution": market_types
    }

def analyze_single_symbol(symbol: str, interval: str = "60") -> Dict[str, Any]:
    """Analyze a single symbol and return the enhanced signal dictionary"""
    raw_analyses = scan_multiple_symbols([symbol], interval)
    if not raw_analyses:
        logger.warning(f"No analysis found for {symbol}")
        return {}

    analysis = raw_analyses[0]
    analysis["score"] = calculate_signal_score(analysis)
    enhanced = enhance_signal(analysis)
    
    # Save to DB with exchange
    signal_obj = Signal(
        symbol=str(enhanced.get("symbol") or "UNKNOWN"),
        interval=interval,
        signal_type=str(enhanced.get("signal_type", "BUY")),
        side=str(enhanced.get("side", "BUY")),
        score=float(enhanced.get("score", 0)),
        entry=float(enhanced.get("entry") or 0),
        sl=float(enhanced.get("sl") or 0),
        tp=float(enhanced.get("tp") or 0),
        trail=float(enhanced.get("trail") or 0),
        liquidation=float(enhanced.get("liquidation") or 0),
        leverage=int(enhanced.get("leverage", 10)),
        margin_usdt=float(enhanced.get("margin_usdt") or 0),
        market=str(enhanced.get("market", "Unknown")),
        indicators=enhanced.get("indicators", {}),
        exchange=EXCHANGE,
        created_at=enhanced.get("created_at") or datetime.now(timezone.utc)
    )
    db_manager.add_signal(signal_obj)

    return enhanced

# Run Standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # For standalone testing
    symbols = get_symbols(limit=20)
    signals = generate_signals(symbols, interval="60", top_n=5)
    summary = get_signal_summary(signals)
    logger.info(f"Signal Summary: {summary}")

    if signals:
        send_all_notifications(signals)