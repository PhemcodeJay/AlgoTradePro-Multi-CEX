import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# === CORRECT IMPORTS FROM indicators.py ===
from indicators import (
    analyze_symbol,      # ← Uses the real one from indicators.py
    fetch_klines,
    scan_symbols,
    get_top_symbols,
    calculate_indicators
)

from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import db as db_manager

# Import clients
from binance_client import BinanceClient
from bybit_client import BybitClient

logger = get_logger(__name__)

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()


# --------------------------------------------------------------------------- #
# Helper: Get client instance
# --------------------------------------------------------------------------- #
def get_client(exchange: str):
    exchange = exchange.lower()
    if exchange == "binance":
        return BinanceClient()
    elif exchange == "bybit":
        return BybitClient()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")


# --------------------------------------------------------------------------- #
# Core: Generate signals
# --------------------------------------------------------------------------- #
def generate_signals(
    exchange: str,
    timeframe: str = '1h',
    max_symbols: int = 50,
    user_id: Optional[int] = None
) -> List[Dict]:
    exchange = exchange.lower()
    if exchange not in ("binance", "bybit"):
        logger.error(f"Invalid exchange: {exchange}")
        return []

    try:
        logger.info(f"Generating signals | {exchange.upper()} | {timeframe} | max={max_symbols}")

        # 1. Get top symbols
        symbols = get_top_symbols(exchange, max_symbols * 2)
        if not symbols:
            logger.warning(f"No symbols for {exchange}")
            return []

        # 2. Parallel analysis
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(analyze_symbol, exchange, sym, timeframe): sym
                for sym in symbols
            }
            raw_results = []
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=30)
                    if result and result.get('side') != 'HOLD':
                        raw_results.append(result)
                except TimeoutError:
                    logger.warning(f"Timeout: {symbol}")
                except Exception as e:
                    logger.error(f"Analysis failed {symbol}: {e}")

        if not raw_results:
            logger.info("No signals generated.")
            return []

        # 3. ML Filter
        try:
            ml_filter = MLFilter(user_id=user_id, exchange=exchange)
            filtered_signals = ml_filter.filter_signals(raw_results)
        except Exception as e:
            logger.warning(f"ML filter failed: {e} → using raw")
            filtered_signals = raw_results

        # 4. Sort & limit
        sorted_signals = sorted(
            filtered_signals,
            key=lambda x: x.get('score', 0),
            reverse=True
        )[:max_symbols]

        # 5. Ensure required fields
        for sig in sorted_signals:
            sig.setdefault('signal_type', 'rsi_sma_vol')
            sig.setdefault('trail', 0.0)
            sig.setdefault('liquidation', 0.0)
            sig.setdefault('margin_usdt', 10.0)
            sig.setdefault('indicators', {})

        # 6. Save to DB
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

        logger.info(f"Generated {len(sorted_signals)} | Saved {saved} | {exchange.upper()}")
        return sorted_signals

    except Exception as e:
        logger.error(f"generate_signals failed: {e}", exc_info=True)
        return []


# --------------------------------------------------------------------------- #
# Utility: Convert numpy types
# --------------------------------------------------------------------------- #
def convert_np_types(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, (np.float64, np.float32, np.float_)):
        return float(data)
    elif isinstance(data, (np.int64, np.int32, np.int_)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


# --------------------------------------------------------------------------- #
# Summary: Signal stats
# --------------------------------------------------------------------------- #
def get_signal_summary(exchange: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    try:
        signals = db_manager.get_signals(limit=100, exchange=exchange.lower(), user_id=user_id)
        buy = len([s for s in signals if getattr(s, 'side', '').lower() == 'buy'])
        sell = len([s for s in signals if getattr(s, 'side', '').lower() == 'sell'])
        last = None
        if signals:
            times = [s.created_at for s in signals if hasattr(s, 'created_at') and s.created_at]
            if times:
                last = max(datetime.fromisoformat(t.replace('Z', '+00:00')) for t in times)
        return {
            'exchange': exchange.lower(),
            'total_signals': len(signals),
            'buy_signals': buy,
            'sell_signals': sell,
            'last_signal_time': last.isoformat() if last else None
        }
    except Exception as e:
        logger.error(f"get_signal_summary error: {e}")
        return {
            'exchange': exchange.lower(),
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'last_signal_time': None
        }

# --------------------------------------------------------------------------- #
# Core: Analyze a single symbol
# --------------------------------------------------------------------------- #
def analyze_single_symbol(
    exchange: str,
    symbol: str,
    timeframe: str = '1h'
) -> Optional[Dict]:
    """
    Analyze a single symbol and return a signal (or None).
    Reuses the same logic as generate_signals but for one symbol.
    """
    exchange = exchange.lower()
    client = get_client(exchange)

    try:
        logger.info(f"Analyzing single symbol: {symbol} | {exchange} | {timeframe}")

        # Fetch klines
        df = fetch_klines(exchange, symbol, timeframe, limit=100)
        if df is None or df.empty:
            logger.warning(f"No data for {symbol}")
            return None

        # Calculate indicators
        indicators = calculate_indicators(df)
        if indicators is None:
            return None

        # Use analyze_symbol logic (already imported)
        result = analyze_symbol(exchange, symbol, timeframe)
        if not result or result.get('side') == 'HOLD':
            return None

        # Ensure required fields
        result.setdefault('signal_type', 'rsi_sma_vol')
        result.setdefault('trail', 0.0)
        result.setdefault('liquidation', 0.0)
        result.setdefault('margin_usdt', 10.0)
        result.setdefault('indicators', indicators)

        # Convert numpy types
        result['indicators'] = convert_np_types(result['indicators'])

        logger.info(f"Signal generated: {symbol} → {result['side']}")
        return result

    except Exception as e:
        logger.error(f"analyze_single_symbol failed for {symbol}: {e}", exc_info=True)
        return None
    
# --------------------------------------------------------------------------- #
# Main: Test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for exch in ["binance", "bybit"]:
        logger.info(f"\n{'='*60}\n TESTING {exch.upper()} \n{'='*60}")
        signals = generate_signals(exchange=exch, timeframe="1h", max_symbols=3)
        summary = get_signal_summary(exch)

        logger.info(f"Summary ({exch}): {summary}")
        if signals:
            top = signals[0]
            logger.info(f"TOP: {top['symbol']} → {top['side']} | Score: {top['score']}")
            send_all_notifications(signals)