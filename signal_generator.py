import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from indicators import (
    analyze_single_symbol, fetch_klines, scan_multiple_symbols,
    get_top_symbols, calculate_indicators
)
from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import db as db_manager

# Import clients
from binance_client import BinanceClient
from bybit_client import BybitClient

logger = get_logger(__name__)

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

# --------------------------------------------------------------------------- #
# Helper: Get correct client instance
# --------------------------------------------------------------------------- #
def get_client(exchange: str):
    """Factory to get BinanceClient or BybitClient"""
    if exchange.lower() == "binance":
        return BinanceClient()
    elif exchange.lower() == "bybit":
        return BybitClient()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

# --------------------------------------------------------------------------- #
# Core: Generate signals for any exchange
# --------------------------------------------------------------------------- #
def generate_signals(
    exchange: str,
    timeframe: str = '1h',
    max_symbols: int = 10,
    user_id: Optional[int] = None
) -> List[Dict]:
    """
    Generate high-confidence trading signals for Binance or Bybit.
    """
    exchange = exchange.lower()
    if exchange not in ("binance", "bybit"):
        logger.error(f"Invalid exchange: {exchange}")
        return []

    try:
        logger.info(f"Generating signals for {exchange.upper()} | timeframe={timeframe} | max_symbols={max_symbols}")

        # 1. Get top symbols
        symbols = get_top_symbols(exchange, max_symbols * 2)  # fetch extra for filtering
        if not symbols:
            logger.warning(f"No symbols returned for {exchange}")
            return []

        # 2. Analyze in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(analyze_single_symbol, exchange, sym, timeframe): sym
                for sym in symbols
            }
            raw_results = []
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result(timeout=30)
                    if result and result.get('side') != 'HOLD':
                        raw_results.append(result)
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.error(f"Error analyzing {symbol}: {e}")

        if not raw_results:
            logger.info(f"No valid signals generated for {exchange}")
            return []

        # 3. Apply ML Filter (with exchange & user context)
        try:
            ml_filter = MLFilter(user_id=user_id, exchange=exchange)
            filtered_signals = ml_filter.filter_signals(raw_results)
        except Exception as e:
            logger.warning(f"ML filter failed, skipping: {e}")
            filtered_signals = raw_results

        # 4. Sort by score & limit
        sorted_signals = sorted(
            filtered_signals,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:max_symbols]

        # 5. Save to DB
        saved = 0
        for sig in sorted_signals:
            db_signal = {
                'symbol': str(sig.get('symbol', '')),
                'interval': str(timeframe),
                'signal_type': str(sig.get('signal_type', 'neutral')),
                'side': str(sig.get('side', 'HOLD')).upper(),
                'score': float(sig.get('score', 0.0)),
                'entry': float(sig.get('entry', 0.0)),
                'sl': float(sig.get('sl', 0.0)),
                'tp': float(sig.get('tp', 0.0)),
                'trail': float(sig.get('trail', 0.0)),
                'liquidation': float(sig.get('liquidation', 0.0)),
                'leverage': int(sig.get('leverage', 1)),
                'margin_usdt': float(sig.get('margin_usdt', 0.0)),
                'market': 'futures',
                'indicators': convert_np_types(sig.get('indicators', {})),
                'exchange': exchange,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            if user_id is not None:
                db_signal['user_id'] = user_id

            if db_manager.add_signal(db_signal):
                saved += 1

        logger.info(f"Generated {len(sorted_signals)} signals | Saved {saved} to DB | Exchange: {exchange.upper()}")
        return sorted_signals

    except Exception as e:
        logger.error(f"Critical error in generate_signals({exchange}): {e}", exc_info=True)
        return []


# --------------------------------------------------------------------------- #
# Utility: Convert numpy types to native Python
# --------------------------------------------------------------------------- #
def convert_np_types(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, np.int64):
        return int(data)
    return data


# --------------------------------------------------------------------------- #
# Summary: Recent signal stats
# --------------------------------------------------------------------------- #
def get_signal_summary(exchange: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    try:
        signals = db_manager.get_signals(limit=100, exchange=exchange.lower(), user_id=user_id)
        buy = len([s for s in signals if s.side.lower() == 'buy'])
        sell = len([s for s in signals if s.side.lower() == 'sell'])
        last = max((s.created_at for s in signals), default=None)
        return {
            'exchange': exchange,
            'total_signals': len(signals),
            'buy_signals': buy,
            'sell_signals': sell,
            'last_signal_time': last.isoformat() if last else None
        }
    except Exception as e:
        logger.error(f"Error in get_signal_summary: {e}")
        return {'exchange': exchange, 'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0, 'last_signal_time': None}


# --------------------------------------------------------------------------- #
# Optional: Manual single-symbol analysis (for testing)
# --------------------------------------------------------------------------- #
def analyze_symbol(
    exchange: str,
    symbol: str,
    timeframe: str = '1h'
) -> Optional[Dict]:
    """Analyze one symbol on Binance or Bybit"""
    try:
        df = fetch_klines(exchange, symbol, timeframe, limit=200)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for {symbol} on {exchange}")
            return None

        df = calculate_indicators(df)
        if df['close'].isna().any():
            return None

        latest = df.iloc[-1]
        signal = {
            'symbol': symbol,
            'exchange': exchange,
            'interval': timeframe,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'price': float(latest['close']),
            'indicators': {
                'rsi': float(latest['rsi']),
                'sma20': float(latest['sma20']),
                'sma50': float(latest['sma50']),
                'atr': float(latest['atr'])
            }
        }

        # Simple logic (can be expanded)
        if latest['rsi'] > 70 and latest['sma20'] < latest['sma50']:
            signal.update({'side': 'sell', 'signal_type': 'bearish_rsi_sma'})
        elif latest['rsi'] < 30 and latest['sma20'] > latest['sma50']:
            signal.update({'side': 'buy', 'signal_type': 'bullish_rsi_sma'})
        else:
            return None

        signal['score'] = 0.7 if signal['side'] == 'buy' and latest['rsi'] < 25 else 0.6
        signal['score'] = 0.7 if signal['side'] == 'sell' and latest['rsi'] > 75 else signal['score']

        # Add SL/TP
        price = signal['price']
        atr = latest['atr']
        if signal['side'] == 'buy':
            signal['sl'] = price - 2 * atr
            signal['tp'] = price + 3 * atr
        else:
            signal['sl'] = price + 2 * atr
            signal['tp'] = price - 3 * atr

        signal['leverage'] = 10
        signal['margin_usdt'] = 3.0

        return signal
    except Exception as e:
        logger.error(f"Error analyzing {symbol} on {exchange}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Main entry (for testing)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test both exchanges
    for exch in ["binance", "bybit"]:
        logger.info(f"\n{'='*60}\n TESTING {exch.upper()} \n{'='*60}")
        signals = generate_signals(exchange=exch, timeframe="60", max_symbols=3)
        summary = get_signal_summary(exch)

        logger.info(f"Signal Summary ({exch}): {summary}")
        if signals:
            send_all_notifications(signals)