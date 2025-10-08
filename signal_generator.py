import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from indicators import scan_multiple_symbols, get_top_symbols
from ml import MLFilter
from notifications import send_all_notifications
from db import Signal, db_manager
from logging_config import get_logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        'created_at': datetime.now(timezone.utc)
    })
    return enhanced

def generate_signals(exchange: str = "binance", timeframe: str = "1h", max_symbols: int = 50, top_n: int = 20) -> List[Dict[str, Any]]:
    """Generate and store trading signals for multiple symbols"""
    try:
        # Step 1: Get top symbols
        symbols = get_top_symbols(exchange, max_symbols)
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
        top_signals = scored[:top_n]
        enhanced_signals = [enhance_signal(s) for s in top_signals]

        # Step 5: Filter using ML model
        ml_filter = MLFilter()
        filtered_signals = ml_filter.filter_signals(enhanced_signals, threshold=0.6)

        # Step 6: Save to DB
        for f in filtered_signals:
            signal_obj = Signal(
                symbol=str(f.get("symbol", "UNKNOWN")),
                interval=timeframe,
                signal_type=str(f.get("signal_type", "BUY")),
                side=str(f.get("side", "BUY")),
                score=float(f.get("score", 0)),
                entry=float(f.get("entry", 0)),
                sl=float(f.get("sl", 0)),
                tp=float(f.get("tp", 0)),
                trail=float(f.get("trail", 0)),
                liquidation=float(f.get("liquidation", 0)),
                leverage=int(f.get("leverage", 10)),
                margin_usdt=float(f.get("margin_usdt", 0)),
                market=str(f.get("market", "Unknown")),
                indicators=f.get("indicators", {}),
                exchange=exchange,
                created_at=f.get("created_at") or datetime.now(timezone.utc)
            )
            db_manager.add_signal(signal_obj)

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
        exchange=exchange,
        created_at=enhanced.get("created_at") or datetime.now(timezone.utc)
    )
    db_manager.add_signal(signal_obj)
    return enhanced

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exchange = "binance"
    symbols = get_symbols(exchange, limit=20)
    signals = generate_signals(exchange, symbols, interval="60", top_n=5)
    summary = get_signal_summary(signals)
    logger.info(f"Signal Summary: {summary}")
    if signals:
        send_all_notifications(signals)