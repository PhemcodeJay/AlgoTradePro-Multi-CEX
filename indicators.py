# indicators.py (fixed data fetching issues, removed env dependency, pass exchange param)
import ccxt
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from logging_config import get_logger
from binance_client import BinanceClient
from bybit_client import BybitClient

logger = get_logger(__name__)

# TRADING_MODE kept for logging (not critical for exchange)
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()  # Used for logging, but data is always real

def get_client(exchange: str) -> Any:
    """Get the appropriate client based on exchange"""
    exchange = exchange.lower()
    if exchange == "binance":
        return BinanceClient()
    elif exchange == "bybit":
        return BybitClient()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

def get_top_symbols(exchange: str, limit: int = 50) -> List[str]:
    """Get top USDT trading pairs by volume using custom client"""
    try:
        client = get_client(exchange)
        # Use underlying CCXT exchange for tickers
        exchange_client = client.exchange
        tickers = exchange_client.fetch_tickers()
        
        # Filter USDT pairs
        usdt_pairs = {symbol: ticker for symbol, ticker in tickers.items() 
                      if symbol.endswith('/USDT') and ticker.get('quoteVolume', 0) > 0}
        
        # Sort by quote volume
        sorted_pairs = sorted(usdt_pairs.items(), 
                              key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
        
        # Convert to symbol format (remove /USDT)
        symbols = [symbol.replace('/USDT', 'USDT') for symbol, _ in sorted_pairs[:limit]]
        
        logger.info(f"Fetched top {len(symbols)} USDT symbols from {exchange}")
        return symbols
        
    except Exception as e:
        logger.error(f"Error fetching top symbols from {exchange}: {e}")
        # Fallback symbols
        return ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]

def fetch_klines(exchange: str, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch klines/candles for a symbol using custom client"""
    try:
        client = get_client(exchange)
        
        # Convert timeframe if needed
        if timeframe == '60':
            timeframe = '1h'
        
        # Fetch using custom client's get_klines
        klines = client.get_klines(symbol, timeframe, limit)
        
        if not klines:
            logger.warning(f"No data received for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)  # Remove rows with NaN values
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        return None

def calculate_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average"""
    return df['close'].astype('float64').rolling(window=period).mean()

def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return df['close'].astype('float64').ewm(span=period, adjust=False).mean()

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = df['close'].astype('float64').diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD"""
    ema_fast = calculate_ema(df, fast)
    ema_slow = calculate_ema(df, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({'macd': macd, 'macd_signal': macd_signal, 'macd_histogram': macd_hist})

def calculate_bb(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(df, period)
    std = df['close'].astype('float64').rolling(window=period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return pd.DataFrame({'bb_upper': upper, 'bb_lower': lower, 'bb_middle': sma})

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate volatility as std dev of returns"""
    returns = df['close'].astype('float64').pct_change()
    return returns.rolling(window=period).std()

def calculate_trend_score(df: pd.DataFrame) -> pd.Series:
    """Calculate simple trend score based on SMAs"""
    sma_short = calculate_sma(df, 20)
    sma_long = calculate_sma(df, 50)
    diff = sma_short - sma_long
    return pd.Series(np.where(diff > 0, 1, np.where(diff < 0, -1, 0)), index=diff.index)

def calculate_volume_ratio(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """Calculate volume ratio"""
    vol_short = df['volume'].rolling(window=short_period).mean()
    vol_long = df['volume'].rolling(window=long_period).mean()
    return vol_short / vol_long

def analyze_symbol(exchange: str, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
    """Analyze a single symbol"""
    try:
        df = fetch_klines(exchange, symbol, timeframe)
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return {}
        
        # Calculate indicators
        sma_20 = calculate_sma(df, 20)
        sma_200 = calculate_sma(df, 200)
        ema_21 = calculate_ema(df, 21)
        ema_9 = calculate_ema(df, 9)
        rsi = calculate_rsi(df)
        macd_data = calculate_macd(df)
        bb_data = calculate_bb(df)
        atr = calculate_atr(df)
        volatility = calculate_volatility(df)
        trend_score = calculate_trend_score(df)
        volume_ratio = calculate_volume_ratio(df)
        
        # Price changes
        price_change_1h = df['close'].pct_change(periods=1).iloc[-1] * 100 if len(df) > 1 else 0
        price_change_4h = df['close'].pct_change(periods=4).iloc[-1] * 100 if len(df) > 4 else 0
        price_change_24h = df['close'].pct_change(periods=24).iloc[-1] * 100 if len(df) > 24 else 0
        
        current_price = df['close'].iloc[-1]
        latest = df.index[-1]
        
        # Determine signal_type and side
        macd_hist = macd_data['macd_histogram']
        if rsi.iloc[-1] < 30 and macd_hist.iloc[-1] > 0 and current_price > sma_20.iloc[-1]:
            signal_type = 'bullish'
            side = 'BUY'
        elif rsi.iloc[-1] > 70 and macd_hist.iloc[-1] < 0 and current_price < sma_20.iloc[-1]:
            signal_type = 'bearish'
            side = 'SELL'
        else:
            signal_type = 'neutral'
            side = 'HOLD'
            return {}  # Skip neutral
        
        # Base score
        base_score = 0
        
        # RSI factor
        if rsi.iloc[-1] < 30 or rsi.iloc[-1] > 70:
            base_score += 20
        
        # MACD factor
        if macd_hist.iloc[-1] > 0 and macd_data['macd'].iloc[-1] > macd_data['macd_signal'].iloc[-1]:
            base_score += 15
        elif macd_hist.iloc[-1] < 0 and macd_data['macd'].iloc[-1] < macd_data['macd_signal'].iloc[-1]:
            base_score += 15
        
        # BB squeeze
        bb_width = (bb_data['bb_upper'].iloc[-1] - bb_data['bb_lower'].iloc[-1]) / bb_data['bb_middle'].iloc[-1]
        if bb_width < 0.05:
            base_score += 10
        
        # Trend factor
        if trend_score.iloc[-1] != 0:
            base_score += 5
        
        # Volume factor
        if volume_ratio.iloc[-1] > 1.2:
            base_score += 10
        
        # Volatility factor
        vol_val = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0
        if 0.01 <= vol_val <= 0.05:
            base_score += 5
        
        indicators = {
            'price': current_price,
            'sma_20': float(sma_20.iloc[-1]),
            'sma_200': float(sma_200.iloc[-1]),
            'ema_21': float(ema_21.iloc[-1]),
            'ema_9': float(ema_9.iloc[-1]),
            'rsi': float(rsi.iloc[-1]),
            'macd': float(macd_data['macd'].iloc[-1]),
            'macd_signal': float(macd_data['macd_signal'].iloc[-1]),
            'macd_histogram': float(macd_data['macd_histogram'].iloc[-1]),
            'bb_upper': float(bb_data['bb_upper'].iloc[-1]),
            'bb_middle': float(bb_data['bb_middle'].iloc[-1]),
            'bb_lower': float(bb_data['bb_lower'].iloc[-1]),
            'atr': float(atr.iloc[-1]),
            'volume_ratio': float(volume_ratio.iloc[-1]),
            'volatility': float(vol_val),
            'trend_score': float(trend_score.iloc[-1]) if not pd.isna(trend_score.iloc[-1]) else 0,
            'price_change_1h': price_change_1h,
            'price_change_4h': price_change_4h,
            'price_change_24h': price_change_24h
        }
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'side': side,
            'score': base_score,
            'indicators': indicators,
            'timestamp': latest.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {}

def scan_multiple_symbols(exchange: str, symbols: List[str], timeframe: str = '1h', max_workers: int = 5) -> List[Dict[str, Any]]:
    """Scan multiple symbols in parallel"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(analyze_symbol, exchange, symbol, timeframe): symbol 
            for symbol in symbols
        }
        
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
                else:
                    logger.warning(f"No result for {symbol}")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
    
    logger.info(f"Analyzed {len(results)}/{len(symbols)} symbols")
    return results

def get_market_overview(exchange: str) -> Dict[str, Any]:
    """Get general market overview"""
    try:
        client = get_client(exchange)
        exchange_client = client.exchange
        
        btc_ticker = exchange_client.fetch_ticker('BTC/USDT')
        eth_ticker = exchange_client.fetch_ticker('ETH/USDT')
        
        return {
            'btc_price': btc_ticker['last'],
            'btc_change_24h': btc_ticker['percentage'],
            'eth_price': eth_ticker['last'],
            'eth_change_24h': eth_ticker['percentage'],
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        return {}

if __name__ == "__main__":
    # Test the indicators
    exchange = "binance"  # Default for standalone testing
    symbols = get_top_symbols(exchange, 5)
    results = scan_multiple_symbols(exchange, symbols)
    
    for result in results:
        print(f"{result['symbol']}: Score {result['score']}, Side: {result['side']}")