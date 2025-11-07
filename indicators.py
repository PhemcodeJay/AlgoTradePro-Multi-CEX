# indicators.py - NO API KEYS REQUIRED - WORKS 2025
import pandas as pd
import numpy as np
import time
import requests
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PUBLIC ENDPOINTS (NO AUTH NEEDED) ===
BINANCE_BASE = "https://api.binance.com"
BYBIT_BASE = "https://api.bybit.com"

SAFE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT"
]

def rate_limited_request(url: str, params: dict = None, max_retries: int = 3) -> Optional[dict]:
    for i in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 2 ** i
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                logger.debug(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            logger.debug(f"Request failed: {e}")
            time.sleep(1)
    return None

# === GET TOP SYMBOLS (PUBLIC) ===
def get_top_symbols(exchange: str, limit: int = 50) -> List[str]:
    try:
        if exchange == "binance":
            data = rate_limited_request(f"{BINANCE_BASE}/api/v3/ticker/24hr")
            if not data:
                raise Exception("No data")
            symbols = [
                d['symbol'] for d in data
                if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 1e7
            ]
            symbols.sort(key=lambda s: next((d['quoteVolume'] for d in data if d['symbol'] == s), 0), reverse=True)
            return symbols[:limit]

        elif exchange == "bybit":
            data = rate_limited_request(f"{BYBIT_BASE}/v5/market/tickers", {"category": "linear"})
            if not data or data.get("retCode") != 0:
                raise Exception(f"Bybit error: {data}")
            symbols = [
                item["symbol"] for item in data["result"]["list"]
                if item["symbol"].endswith("USDT") and float(item.get("turnover24h", 0)) > 1e7
            ]
            symbols.sort(
                key=lambda s: float(next(
                    (item["turnover24h"] for item in data["result"]["list"] if item["symbol"] == s),
                    0
                )),
                reverse=True
            )
            return symbols[:limit]
    except Exception as e:
        logger.warning(f"Top symbols failed ({exchange}): {e} → using SAFE_SYMBOLS")
        return SAFE_SYMBOLS[:limit]

# === FETCH KLINES (PUBLIC ENDPOINTS ONLY) ===
def fetch_klines(exchange: str, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
    for attempt in range(3):
        try:
            if exchange == "binance":
                params = {
                    "symbol": symbol,
                    "interval": timeframe,
                    "limit": limit
                }
                raw = rate_limited_request(f"{BINANCE_BASE}/api/v3/klines", params)
                if not raw or len(raw) < 50:
                    raise Exception("Insufficient data")

                df = pd.DataFrame(raw, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'tb_base', 'tb_quote', 'ignore'
                ])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            elif exchange == "bybit":
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": timeframe.replace('h', '60').replace('m', ''),
                    "limit": limit
                }
                raw = rate_limited_request(f"{BYBIT_BASE}/v5/market/kline", params)
                if not raw or raw.get("retCode") != 0 or not raw.get("result", {}).get("list"):
                    raise Exception("Invalid response")
                
                klines = raw["result"]["list"]
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='last')].sort_index()
            df = df.dropna()

            if len(df) >= 50:
                return df

        except Exception as e:
            logger.debug(f"Kline retry {attempt+1}/3 for {symbol} ({exchange}): {e}")
            time.sleep(2 ** attempt)

    logger.error(f"Kline failed permanently: {symbol} on {exchange}")
    return None

# === INDICATORS ===
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    return df.dropna()

# === SIGNAL LOGIC ===
def analyze_symbol(exchange: str, symbol: str, timeframe: str = '1h') -> Optional[Dict[str, Any]]:
    df = fetch_klines(exchange, symbol, timeframe)
    if df is None or len(df) < 100:
        return None

    df = calculate_indicators(df)
    if df.empty:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    conditions = []
    side = "HOLD"

    # RSI
    if latest['rsi'] < 30:
        score += 40
        conditions.append("RSI_OVERSOLD")
    elif latest['rsi'] > 70:
        score += 40
        conditions.append("RSI_OVERBOUGHT")

    # SMA Trend
    if latest['sma20'] > latest['sma50']:
        score += 30
        conditions.append("BULLISH_TREND")
    else:
        score += 30
        conditions.append("BEARISH_TREND")

    # Volatility
    if latest['atr'] > latest['close'] * 0.006:
        score += 20
        conditions.append("HIGH_VOL")

    # Entry logic
    if score >= 80 and "RSI_OVERSOLD" in conditions and "BULLISH_TREND" in conditions:
        side = "BUY"
    elif score >= 80 and "RSI_OVERBOUGHT" in conditions and "BEARISH_TREND" in conditions:
        side = "SELL"

    if side == "HOLD":
        return None

    price = latest['close']
    return {
        "symbol": symbol,
        "side": side,
        "score": int(score),
        "price": float(price),
        "rsi": round(latest['rsi'], 2),
        "sma20": round(latest['sma20'], 4),
        "sma50": round(latest['sma50'], 4),
        "atr": round(latest['atr'], 4),
        "conditions": conditions,
        "timestamp": df.index[-1].isoformat(),
        "tp": round(price * 1.02 if side == "BUY" else price * 0.98, 4),
        "sl": round(price * 0.98 if side == "BUY" else price * 1.02, 4),
        "leverage": 5
    }

# === SCANNER ===
def scan_symbols(exchange: str, limit: int = 20, timeframe: str = '1h') -> List[Dict]:
    symbols = get_top_symbols(exchange, limit * 2)[:limit * 2]
    results = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(analyze_symbol, exchange, symbol, timeframe): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            try:
                if result := future.result(timeout=15):
                    results.append(result)
            except:
                pass
            time.sleep(0.2)  # Be gentle on APIs

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Scan complete: {len(results)} signals from {len(symbols)} symbols")
    return results

# === TEST ===
if __name__ == "__main__":
    exchange = "bybit"  # or "binance"
    signals = scan_symbols(exchange, limit=15)
    
    print(f"\nTOP SIGNALS on {exchange.upper()} ({time.strftime('%Y-%m-%d %H:%M')})\n")
    for s in signals[:10]:
        print(f"{s['symbol']:>10} | {s['side']:^4} | Score: {s['score']} | RSI: {s['rsi']:>5} | Price: {s['price']:>10} | {', '.join(s['conditions'])}")