import logging
from typing import List, Dict, Any, Union
from datetime import datetime, timezone
from indicators import scan_multiple_symbols, get_top_symbols
from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = get_logger(__name__)

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

def get_symbols(exchange: str, limit: int = 50) -> List[str]:
    usdt_symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT",
        "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT"
    ]
    if exchange == "bybit":
        usdt_symbols.extend(["NEARUSDT", "ALGOUSDT", "TRXUSDT"])
    elif exchange == "binance":
        usdt_symbols.extend(["TRXUSDT", "VETUSDT", "ALGOUSDT"])
    usdt_symbols = list(dict.fromkeys(usdt_symbols))
    try:
        top_symbols = get_top_symbols(exchange, limit)
        usdt_symbols = [s for s in top_symbols if s in usdt_symbols][:limit]
        logger.info(f"Retrieved {len(usdt_symbols)} USDT trading pairs for {exchange} via API")
    except Exception as e:
        logger.warning(f"Error sorting symbols by volume for {exchange}: {e}")
        usdt_symbols = usdt_symbols[:limit]
    if not usdt_symbols:
        logger.warning("No USDT symbols available, using fallback list")
        usdt_symbols = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
    logger.info(f"Final {len(usdt_symbols)} USDT trading pairs for {exchange}")
    return sorted(usdt_symbols)

def calculate_signal_score(analysis: Dict[str, Any]) -> float:
    score = float(analysis.get("score", 0))
    indicators = analysis.get("indicators", {})
    if not isinstance(indicators, dict):
        logger.warning("Indicators is not a dictionary")
        return score
    
    rsi = float(indicators.get("rsi", 50))
    if 20 <= rsi <= 30 or 70 <= rsi <= 80:
        score += 10
    elif rsi < 20 or rsi > 80:
        score += 5
    macd_hist = float(indicators.get("macd_histogram", 0))
    if abs(macd_hist) > 0.01:
        score += 8
    vol_ratio = float(indicators.get("volume_ratio", 1))
    if vol_ratio > 2:
        score += 12
    elif vol_ratio > 1.5:
        score += 6
    vol = float(indicators.get("volatility", 0))
    if 0.5 <= vol <= 3:
        score += 5
    elif vol > 5:
        score -= 10
    trend_score = float(indicators.get("trend_score", 0))
    score += trend_score * 3
    return min(100, max(0, score))

# Utility function to convert np.float64 to float recursively
def convert_np_types(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    return data

def enhance_signal(analysis: Dict[str, Any]) -> Dict[str, Any]:
    # Validate analysis
    if not isinstance(analysis, dict):
        logger.error("Analysis must be a dictionary")
        return {}
    
    # Convert and validate indicators
    indicators = convert_np_types(analysis.get("indicators", {}))
    if not isinstance(indicators, dict):
        logger.error("Indicators must be a dictionary")
        return analysis

    price = indicators.get("price", 0)
    atr = indicators.get("atr", 0)
    
    # Validate that price and atr are numbers
    try:
        price = float(price)
        atr = float(atr)
    except (TypeError, ValueError):
        logger.error(f"Invalid price or ATR value: price={price}, atr={atr}")
        return analysis

    if atr <= 0:
        logger.warning(f"ATR is zero or negative for {analysis.get('symbol', 'unknown')}, using default values")
        atr = 0.01 * price  # Fallback to 1% of price

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
    
    margin_usdt = price * 1 / leverage
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
        'created_at': datetime.now(timezone.utc).isoformat()
    })
    return convert_np_types(enhanced)

def generate_signals(exchange: str = "binance", timeframe: str = "1h", max_symbols: int = 50) -> List[Dict[str, Any]]:
    """Generate trading signals for multiple symbols"""
    try:
        # Step 1: Get top symbols
        symbols = get_symbols(exchange, max_symbols)
        logger.info(f"Generating signals for {len(symbols)} symbols on {exchange} ({timeframe})")

        signals = []
        # Step 2: Analyze each symbol concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(analyze_single_symbol, exchange, symbol, timeframe): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result and result.get('symbol'):
                        signals.append(result)
                        logger.debug(f"Signal generated for {symbol}")
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

        # Step 3: Score and rank signals
        for sig in signals:
            sig['score'] = calculate_signal_score(sig)
        scored = sorted(signals, key=lambda x: x['score'], reverse=True)

        # Step 4: Select top signals and enhance them
        top_signals = scored[:max_symbols]
        enhanced_signals = [enhance_signal(s) for s in top_signals]

        # Step 5: Filter using ML model
        ml_filter = MLFilter()
        filtered_signals = ml_filter.filter_signals(enhanced_signals, threshold=0.6)

        logger.info(f"✅ Generated {len(filtered_signals)} filtered signals out of {len(signals)} analyzed.")
        return filtered_signals

    except Exception as e:
        logger.error(f"An error occurred during signal generation: {e}")
        return []

def get_signal_summary(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
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

def analyze_single_symbol(exchange: str, symbol: str, interval: str = "60") -> Dict[str, Any]:
    raw_analyses = scan_multiple_symbols(exchange, [symbol], interval)
    if not raw_analyses:
        logger.warning(f"No analysis found for {symbol}")
        return {}
    analysis = raw_analyses[0]
    analysis["score"] = calculate_signal_score(analysis)
    enhanced = enhance_signal(analysis)
    return enhanced

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exchange = "binance"
    signals = generate_signals(exchange, timeframe="60", max_symbols=5)
    summary = get_signal_summary(signals)
    logger.info(f"Signal Summary: {summary}")
    if signals:
        send_all_notifications(signals)