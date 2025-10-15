import ccxt
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from logging_config import get_trading_logger
from binance_client import BinanceClient
from bybit_client import BybitClient

logger = get_trading_logger(__name__)

_client_cache = {}
def get_client(exchange: str) -> Any:
    """Get or reuse the appropriate client based on exchange"""
    exchange = exchange.lower()
    if exchange not in _client_cache:
        if exchange == "binance":
            _client_cache[exchange] = BinanceClient()
        elif exchange == "bybit":
            _client_cache[exchange] = BybitClient()
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    return _client_cache[exchange]

# Utility function to convert np.float64 to float recursively
def convert_np_types(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.float64):
        return float(data)
    return data

def get_top_symbols(exchange: str, limit: int = 50) -> List[str]:
    """Get top USDT trading pairs by volume using custom client"""
    fallback_symbols = [
        "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT",
        "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "MATICUSDT", "SHIBUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT",
        "TRXUSDT", "VETUSDT", "ALGOUSDT"
    ]
    symbols = list(dict.fromkeys(fallback_symbols))[:limit]
    logger.info(f"Using predefined {len(symbols)} USDT symbols for {exchange}")
    return symbols

def fetch_klines(exchange: str, symbol: str, timeframe: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch klines/candles for a symbol using custom client"""
    try:
        client = get_client(exchange)
        timeframe_map = {
            '1h': '60', '4h': '240', '1d': '1440',
            '60': '60', '240': '240', '1440': '1440'
        }
        interval = timeframe_map.get(timeframe, '60')
        
        klines = client.get_klines(symbol, interval, limit)
        logger.debug(f"Fetched {len(klines)} klines for {symbol}: {klines[:2] if klines else []}")
        
        if not klines:
            logger.warning(f"No data received for {symbol}")
            return None
        
        df = pd.DataFrame(klines)
        logger.debug(f"Raw DataFrame for {symbol}: {df.head(2).to_dict()}")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in klines for {symbol}: {df.columns}")
            return None
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Dropping {nan_count} rows with NaN values for {symbol}")
            df.dropna(inplace=True)
        
        if df.empty:
            logger.warning(f"DataFrame empty after cleaning for {symbol}")
            return None
        
        logger.info(f"Processed DataFrame for {symbol} with {len(df)} rows")
        return df
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg or "451" in error_msg or "restricted location" in error_msg:
            logger.warning(f"API access restricted for {symbol}. This may be due to geographical restrictions.")
        else:
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

def calculate_stoch_rsi(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    """Calculate Stochastic RSI"""
    rsi = calculate_rsi(df, period)
    stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min() + 1e-10) * 100
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return pd.DataFrame({'stoch_rsi_k': k, 'stoch_rsi_d': d})

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

def calculate_volume_ratio(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """Calculate volume ratio"""
    vol_short = df['volume'].rolling(window=short_period).mean()
    vol_long = df['volume'].rolling(window=long_period).mean()
    return vol_short / vol_long

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)"""
    high_low = df['high'].astype('float64') - df['low'].astype('float64')
    high_close = abs(df['high'].astype('float64') - df['close'].astype('float64').shift())
    low_close = abs(df['low'].astype('float64') - df['close'].astype('float64').shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def analyze_symbol(exchange: str, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
    """Analyze a single symbol using combined mean reversion and trend-following strategies"""
    try:
        df = fetch_klines(exchange, symbol, timeframe)
        if df is None or len(df) < 200:
            logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
            return {}
        
        # Calculate indicators
        sma_200 = calculate_sma(df, 200)
        ema_9 = calculate_ema(df, 9)
        bb_data = calculate_bb(df, 20)
        stoch_rsi_data = calculate_stoch_rsi(df, 14)
        rsi = calculate_rsi(df, 14)
        macd_data = calculate_macd(df)
        volume_ratio = calculate_volume_ratio(df)
        atr = calculate_atr(df, 14)
        
        # Check for valid indicator values
        if any(pd.isna([
            sma_200.iloc[-1], ema_9.iloc[-1], bb_data['bb_upper'].iloc[-1],
            stoch_rsi_data['stoch_rsi_k'].iloc[-1], rsi.iloc[-1],
            macd_data['macd_histogram'].iloc[-1], volume_ratio.iloc[-1],
            atr.iloc[-1]
        ])):
            logger.warning(f"Invalid indicator values for {symbol}")
            return {}
        
        current_price = df['close'].iloc[-1]
        latest = df.index[-1]
        
        # Mean Reversion Signals
        mr_score = 0
        mr_conditions = []
        
        # Bollinger Bands: Price near lower/upper band
        bb_width = (bb_data['bb_upper'].iloc[-1] - bb_data['bb_lower'].iloc[-1]) / bb_data['bb_middle'].iloc[-1]
        bb_position = (current_price - bb_data['bb_lower'].iloc[-1]) / (bb_data['bb_upper'].iloc[-1] - bb_data['bb_lower'].iloc[-1] + 1e-10)
        if bb_position < 0.1 and bb_width < 0.1:  # Price near lower band, narrow bands
            mr_score += 30
            mr_conditions.append("BB_lower")
        elif bb_position > 0.9 and bb_width < 0.1:  # Price near upper band, narrow bands
            mr_score += 30
            mr_conditions.append("BB_upper")
        
        # Stochastic RSI: Overbought/oversold
        if stoch_rsi_data['stoch_rsi_k'].iloc[-1] < 20 and stoch_rsi_data['stoch_rsi_k'].iloc[-1] > stoch_rsi_data['stoch_rsi_d'].iloc[-1]:
            mr_score += 25
            mr_conditions.append("Stoch_RSI_oversold")
        elif stoch_rsi_data['stoch_rsi_k'].iloc[-1] > 80 and stoch_rsi_data['stoch_rsi_k'].iloc[-1] < stoch_rsi_data['stoch_rsi_d'].iloc[-1]:
            mr_score += 25
            mr_conditions.append("Stoch_RSI_overbought")
        
        # RSI: Overbought/oversold
        if rsi.iloc[-1] < 30:
            mr_score += 20
            mr_conditions.append("RSI_oversold")
        elif rsi.iloc[-1] > 70:
            mr_score += 20
            mr_conditions.append("RSI_overbought")
        
        # Trend-Following Signals
        tf_score = 0
        tf_conditions = []
        
        # 200 SMA and 9 EMA alignment
        if current_price > sma_200.iloc[-1] and ema_9.iloc[-1] > sma_200.iloc[-1]:
            tf_score += 30
            tf_conditions.append("Price_above_SMA200_EMA9")
        elif current_price < sma_200.iloc[-1] and ema_9.iloc[-1] < sma_200.iloc[-1]:
            tf_score += 30
            tf_conditions.append("Price_below_SMA200_EMA9")
        
        # MACD: Trend confirmation
        if macd_data['macd'].iloc[-1] > macd_data['macd_signal'].iloc[-1] and macd_data['macd_histogram'].iloc[-1] > 0:
            tf_score += 25
            tf_conditions.append("MACD_bullish")
        elif macd_data['macd'].iloc[-1] < macd_data['macd_signal'].iloc[-1] and macd_data['macd_histogram'].iloc[-1] < 0:
            tf_score += 25
            tf_conditions.append("MACD_bearish")
        
        # Volume confirmation
        volume_score = 0
        if volume_ratio.iloc[-1] > 1.5:
            volume_score += 20
            mr_conditions.append("High_volume")
            tf_conditions.append("High_volume")
        
        # Combine scores
        total_score = mr_score + tf_score + volume_score
        signal_type = "neutral"
        side = "HOLD"
        
        # Mean Reversion: BUY if oversold, SELL if overbought
        if mr_score >= 50 and "BB_lower" in mr_conditions and ("Stoch_RSI_oversold" in mr_conditions or "RSI_oversold" in mr_conditions):
            signal_type = "mean_reversion_bullish"
            side = "BUY"
        elif mr_score >= 50 and "BB_upper" in mr_conditions and ("Stoch_RSI_overbought" in mr_conditions or "RSI_overbought" in mr_conditions):
            signal_type = "mean_reversion_bearish"
            side = "SELL"
        
        # Trend-Following: BUY if bullish trend, SELL if bearish trend
        elif tf_score >= 50 and "Price_above_SMA200_EMA9" in tf_conditions and "MACD_bullish" in tf_conditions:
            signal_type = "trend_following_bullish"
            side = "BUY"
        elif tf_score >= 50 and "Price_below_SMA200_EMA9" in tf_conditions and "MACD_bearish" in tf_conditions:
            signal_type = "trend_following_bearish"
            side = "SELL"
        
        # Log details for debugging
        logger.debug(f"{symbol} RSI: {rsi.iloc[-1]:.2f}, Stoch_RSI_K: {stoch_rsi_data['stoch_rsi_k'].iloc[-1]:.2f}, "
                     f"MACD_Hist: {macd_data['macd_histogram'].iloc[-1]:.6f}, Price: {current_price:.6f}, "
                     f"SMA200: {sma_200.iloc[-1]:.6f}, EMA9: {ema_9.iloc[-1]:.6f}, "
                     f"BB_Position: {bb_position:.2f}, Volume_Ratio: {volume_ratio.iloc[-1]:.2f}, "
                     f"ATR: {atr.iloc[-1]:.6f}")
        logger.info(f"Signal for {symbol}: {signal_type}, Side: {side}, MR_Score: {mr_score}, TF_Score: {tf_score}, "
                    f"Total_Score: {total_score}, Conditions: {mr_conditions + tf_conditions}")
        
        indicators = {
            'price': float(current_price),
            'sma_200': float(sma_200.iloc[-1]),
            'ema_9': float(ema_9.iloc[-1]),
            'rsi': float(rsi.iloc[-1]),
            'stoch_rsi_k': float(stoch_rsi_data['stoch_rsi_k'].iloc[-1]),
            'stoch_rsi_d': float(stoch_rsi_data['stoch_rsi_d'].iloc[-1]),
            'macd': float(macd_data['macd'].iloc[-1]),
            'macd_signal': float(macd_data['macd_signal'].iloc[-1]),
            'macd_histogram': float(macd_data['macd_histogram'].iloc[-1]),
            'bb_upper': float(bb_data['bb_upper'].iloc[-1]),
            'bb_middle': float(bb_data['bb_middle'].iloc[-1]),
            'bb_lower': float(bb_data['bb_lower'].iloc[-1]),
            'volume_ratio': float(volume_ratio.iloc[-1]),
            'atr': float(atr.iloc[-1]),
            'price_change_1h': float(df['close'].pct_change(periods=1).iloc[-1] * 100 if len(df) > 1 else 0),
            'price_change_4h': float(df['close'].pct_change(periods=4).iloc[-1] * 100 if len(df) > 4 else 0),
            'price_change_24h': float(df['close'].pct_change(periods=24).iloc[-1] * 100 if len(df) > 24 else 0)
        }
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'side': side,
            'score': total_score,
            'mr_score': mr_score,
            'tf_score': tf_score,
            'indicators': indicators,
            'timestamp': latest.isoformat(),
            'entry': float(current_price),
            'tp': float(current_price * 1.02 if side == "BUY" else current_price * 0.98),
            'sl': float(current_price * 0.98 if side == "BUY" else current_price * 1.02),
            'leverage': 1
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {}

def analyze_single_symbol(exchange: str, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
    """Wrapper for analyzing a single symbol"""
    return analyze_symbol(exchange, symbol, timeframe)

def scan_multiple_symbols(exchange: str, symbols: List[str], timeframe: str = '1h', max_workers: int = 5) -> List[Dict[str, Any]]:
    """Scan multiple symbols in parallel"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
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
    return [convert_np_types(result) for result in results]

def get_market_overview(exchange: str) -> Dict[str, Any]:
    """Get general market overview"""
    try:
        client = get_client(exchange)
        exchange_client = client.exchange
        
        btc_ticker = exchange_client.fetch_ticker('BTC/USDT')
        eth_ticker = exchange_client.fetch_ticker('ETH/USDT')
        
        return {
            'btc_price': float(btc_ticker['last']),
            'btc_change_24h': float(btc_ticker['percentage']),
            'eth_price': float(eth_ticker['last']),
            'eth_change_24h': float(eth_ticker['percentage']),
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market overview: {e}")
        return {}

if __name__ == "__main__":
    # Test the indicators
    exchange = "binance"
    symbols = get_top_symbols(exchange, 5)
    results = scan_multiple_symbols(exchange, symbols)
    
    for result in results:
        print(f"{result['symbol']}: Score {result['score']}, Side: {result['side']}")