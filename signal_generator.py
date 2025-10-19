import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone

import pandas as pd
from indicators import analyze_single_symbol, fetch_klines, scan_multiple_symbols, get_top_symbols, calculate_indicators
from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import db as db_manager

logger = get_logger(__name__)

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

def get_symbols(exchange: str, limit: int = 50) -> List[str]:
    """Retrieve trading pairs for the specified exchange"""
    try:
        symbols = get_top_symbols(exchange, limit)
        logger.info(f"Retrieved {len(symbols)} real USDT trading pairs for {exchange} via API")
        return sorted(symbols)
    except Exception as e:
        logger.warning(f"Error fetching symbols for {exchange}: {e}")
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
    
def calculate_signal_score(signal: Dict) -> float:
    """Calculate confidence score for a signal"""
    try:
        rsi = signal['indicators']['rsi']
        score = 0.5
        
        if signal['side'] == 'buy' and rsi < 30:
            score += 0.3
        elif signal['side'] == 'sell' and rsi > 70:
            score += 0.3
            
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.error(f"Error calculating signal score: {e}")
        return 0.5
    
# Utility function to convert np.float64 to float recursively
def convert_np_types(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    return data

def enhance_signal(signal: Dict, latest: pd.Series) -> Dict:
    """Enhance signal with additional parameters"""
    try:
        atr = latest['atr']
        price = signal['price']
        
        signal['entry'] = price
        signal['sl'] = price - (2 * atr) if signal['side'] == 'buy' else price + (2 * atr)
        signal['tp'] = price + (3 * atr) if signal['side'] == 'buy' else price - (3 * atr)
        signal['trail'] = atr
        signal['leverage'] = 10
        signal['margin_usdt'] = 3.0
        signal['market'] = 'futures'
        
        return signal
    except Exception as e:
        logger.error(f"Error enhancing signal: {e}")
        return signal
    
def generate_signals(exchange: str, symbols: List[str], interval: str = '1h', top_n: int = 10, user_id: int = 1) -> List[Dict]:
    """Generate trading signals for multiple symbols"""
    try:
        signals = []
        for symbol in symbols:
            signal = analyze_symbol(exchange, symbol, interval)
            if signal:
                signal['user_id'] = user_id
                signals.append(signal)
        
        ml_filter = MLFilter()
        signals = ml_filter.filter_signals(signals)
        
        signals = sorted(signals, key=lambda x: x['score'], reverse=True)[:top_n]
        
        for signal in signals:
            db_manager.add_signal(signal)
        
        logger.info(f"Generated {len(signals)} signals for {exchange}")
        return signals
    except Exception as e:
        logger.error(f"Error generating signals for {exchange}: {e}")
        return []
    
def get_signal_summary(exchange: str, user_id: int) -> Dict[str, Any]:
    """Get summary of recent signals"""
    try:
        signals = db_manager.get_signals(limit=100, exchange=exchange, user_id=user_id)
        return {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s.side.lower() == 'buy']),
            'sell_signals': len([s for s in signals if s.side.lower() == 'sell']),
            'last_signal_time': max((s.created_at for s in signals), default=None)
        }
    except Exception as e:
        logger.error(f"Error getting signal summary: {e}")
        return {'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0, 'last_signal_time': None}
    

def analyze_symbol(exchange: str, symbol: str, timeframe: str = '1h') -> Optional[Dict]:
    """Analyze a single symbol and generate a trading signal"""
    try:
        df = fetch_klines(exchange, symbol, timeframe, limit=200)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for {symbol} on {exchange}")
            return None

        df = calculate_indicators(df)
        
        if df['close'].isna().any() or df['rsi'].isna().any():
            logger.warning(f"Invalid data for {symbol} on {exchange}")
            return None

        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
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

        if latest['rsi'] > 70 and latest['sma20'] < latest['sma50']:
            signal['side'] = 'sell'
            signal['signal_type'] = 'bearish_crossover_rsi_overbought'
        elif latest['rsi'] < 30 and latest['sma20'] > latest['sma50']:
            signal['side'] = 'buy'
            signal['signal_type'] = 'bullish_crossover_rsi_oversold'
        else:
            return None

        signal['score'] = calculate_signal_score(signal)
        signal = enhance_signal(signal, latest)
        
        return signal

    except Exception as e:
        logger.error(f"Error analyzing {symbol} on {exchange}: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    exchange = "binance"
    signals = generate_signals(exchange, timeframe="60", max_symbols=5)
    summary = get_signal_summary(signals)
    logger.info(f"Signal Summary: {summary}")
    if signals:
        send_all_notifications(signals)