import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging_config import get_trading_logger
from exceptions import APIConnectionException, APIDataException
from binance_client import BinanceClient
from bybit_client import BybitClient

logger = get_trading_logger(__name__)

# === CLIENT CACHE ===
_client_cache = {}
def get_client(exchange: str):
    if exchange not in _client_cache:
        if exchange == "binance":
            _client_cache[exchange] = BinanceClient()
        elif exchange == "bybit":
            _client_cache[exchange] = BybitClient()
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    return _client_cache[exchange]

# === FALLBACK SYMBOLS (SAFE & VALID) ===
SAFE_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "BNBUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT"
]

# === TOP SYMBOLS (WITH RETRY + FALLBACK) ===
def get_top_symbols(exchange: str, limit: int = 50) -> List[str]:
    client = get_client(exchange)
    try:
        symbols = []
        if exchange == "binance":
            tickers = client.exchange.fetch_tickers()
            symbols = [
                s.replace('/', '') for s, t in tickers.items()
                if s.endswith('/USDT') and t.get('quoteVolume', 0) > 0
            ]
        elif exchange == "bybit":
            result = client._make_request("GET", "/v5/market/tickers", {"category": "linear"})
            if result.get("retCode") != 0:
                raise Exception(f"Bybit tickers error: {result.get('retMsg')}")
            symbols = [item["symbol"] for item in result["result"]["list"] if item["symbol"].endswith("USDT")]
        
        # Sort by volume (fallback to safe list if fails)
        if len(symbols) < limit:
            raise Exception("Not enough symbols")
        symbols.sort(
            key=lambda s: client.get_current_price(s) * client._make_request(
                "GET", "/v5/market/tickers", {"category": "linear", "symbol": s}
            )["result"]["list"][0].get("turnover24h", 0) if exchange == "bybit" else 1,
            reverse=True
        )
        return symbols[:limit]

    except Exception as e:
        logger.warning(f"Top symbols failed ({exchange}): {e} → using fallback")
        return SAFE_SYMBOLS[:limit]

# === KLINE FETCH (WITH RETRY + VALIDATION) ===
def fetch_klines(exchange: str, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
    client = get_client(exchange)
    for attempt in range(3):
        try:
            if exchange == "binance":
                raw = client.exchange.fetch_ohlcv(symbol.replace('USDT', '/USDT'), timeframe, limit=limit)
            else:  # bybit
                result = client._make_request("GET", "/v5/market/kline", {
                    "category": "linear", "symbol": symbol, "interval": timeframe, "limit": limit
                })
                if result.get("retCode") != 0 or not result.get("result", {}).get("list"):
                    raise Exception("Invalid kline response")
                raw = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in result["result"]["list"]]

            if not raw or len(raw) < 50:
                raise Exception("Insufficient data")

            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df = df[~df.index.duplicated(keep='last')].sort_index()
            return df.dropna() if len(df) >= 50 else None

        except Exception as e:
            logger.debug(f"Kline retry {attempt+1}/3 for {symbol}: {e}")
            time.sleep(1)
    logger.error(f"Kline failed for {symbol}")
    return None

# === INDICATOR CALCULATIONS (CLEAN & SAFE) ===
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    return df.dropna()

# === SIGNAL ANALYSIS (ROBUST & FAST) ===
def analyze_symbol(exchange: str, symbol: str, timeframe: str = '1h') -> Optional[Dict[str, Any]]:
    df = fetch_klines(exchange, symbol, timeframe)
    if df is None or len(df) < 100:
        return None

    df = calculate_indicators(df)
    if df.empty:
        return None

    p = df['close'].iloc[-1]
    sma20 = df['sma20'].iloc[-1]
    sma50 = df['sma50'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    atr = df['atr'].iloc[-1]

    # === SIMPLE STRATEGY: RSI + SMA CROSS + VOLATILITY FILTER ===
    score = 0
    side = "HOLD"
    conditions = []

    # RSI Oversold/Overbought
    if rsi < 30:
        score += 40
        conditions.append("RSI_oversold")
    elif rsi > 70:
        score += 40
        conditions.append("RSI_overbought")

    # SMA Trend
    if sma20 > sma50:
        score += 30
        conditions.append("SMA_bullish")
    else:
        score += 30
        conditions.append("SMA_bearish")

    # Volatility filter (avoid low ATR)
    if atr > p * 0.005:  # >0.5%
        score += 20
        conditions.append("High_volatility")

    # Final decision
    if score >= 70 and "RSI_oversold" in conditions and "SMA_bullish" in conditions:
        side = "BUY"
    elif score >= 70 and "RSI_overbought" in conditions and "SMA_bearish" in conditions:
        side = "SELL"

    return {
        "symbol": symbol,
        "side": side,
        "score": score,
        "price": float(p),
        "rsi": float(rsi),
        "sma20": float(sma20),
        "sma50": float(sma50),
        "atr": float(atr),
        "conditions": conditions,
        "timestamp": df.index[-1].isoformat(),
        "entry": float(p),
        "tp": float(p * 1.015 if side == "BUY" else p * 0.985),
        "sl": float(p * 0.985 if side == "BUY" else p * 1.015),
        "leverage": 5
    }

# === PARALLEL SCANNER ===
def scan_symbols(exchange: str, symbols: List[str], timeframe: str = '1h') -> List[Dict]:
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(analyze_symbol, exchange, s, timeframe): s for s in symbols}
        for future in as_completed(futures):
            try:
                if result := future.result(timeout=20):
                    results.append(result)
            except:
                pass
    logger.info(f"Scanned {len(results)}/{len(symbols)} symbols")
    return sorted(results, key=lambda x: x["score"], reverse=True)

# === MARKET OVERVIEW ===
def get_market_overview(exchange: str) -> Dict:
    client = get_client(exchange)
    try:
        price = client.get_current_price("BTCUSDT")
        return {"btc_price": price, "timestamp": time.time()}
    except:
        return {"btc_price": 0, "timestamp": time.time()}

# === MAIN TEST ===
if __name__ == "__main__":
    exchange = "bybit"
    symbols = get_top_symbols(exchange, 10)
    results = scan_symbols(exchange, symbols)
    for r in results[:5]:
        print(f"{r['symbol']}: {r['side']} | Score: {r['score']} | RSI: {r['rsi']:.1f}")