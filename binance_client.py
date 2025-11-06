import hashlib
import hmac
import os
import threading
import time
import json
import asyncio
import websockets
from typing import Dict, List, Any, Optional, TYPE_CHECKING, cast
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
import ccxt
import random
from dataclasses import dataclass
from logging_config import get_trading_logger
from exceptions import (
    APIException, APIConnectionException, APIRateLimitException,
    APIAuthenticationException, APITimeoutException, APIDataException,
    APIErrorRecoveryStrategy, create_error_context
)
from dotenv import load_dotenv  # <-- ADD THIS

# Load environment variables
load_dotenv()

if TYPE_CHECKING:
    from db import DatabaseManager, Trade, WalletBalance
else:
    from db import DatabaseManager, Trade, WalletBalance

logger = get_trading_logger('api_client')
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()


@dataclass
class RateLimitInfo:
    requests_per_second: int = 20   # Futures: higher limit
    requests_per_minute: int = 1200
    last_request_time: float = 0.0
    minute_start: float = 0.0
    minute_count: int = 0


class BinanceClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        # === API CREDENTIALS ===
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise APIConnectionException("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env")

        # === FUTURES ONLY ===
        self.account_type = "FUTURES"
        self.base_url = "https://fapi.binance.com"           # FUTURES REST
        self.ws_url = "wss://fstream.binance.com"            # FUTURES WS
        self.session = None
        self.ws_connection = None
        self.loop = None
        self._price_cache: Dict[str, tuple[float, float]] = {}
        self._connected = False
        self._connection_lock = threading.Lock()

        # Rate limiting
        self.rate_limit = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        self.production_rate_limit_interval = 1
        self.production_rate_limit_count = 20
        self._last_reset = datetime.now(timezone.utc)
        self._request_count = 0

        # Stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_successful_request = None

        # === CCXT FUTURES CONFIG ===
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'future',           # FUTURES ONLY
                'adjustForTimeDifference': True,   # FIX -1021
                'recvWindow': 10000,               # 10s tolerance
                'fetchCurrencies': False,          # FIX -2015
            },
        })
        self.exchange.set_sandbox_mode(False)

        # Time sync on init
        try:
            self.exchange.load_time_difference()
            logger.info(f"Time sync successful. Offset: {self.exchange.timeDifference}ms")
        except Exception as e:
            logger.warning(f"Time sync failed at startup: {e}")

        self._initialize_session()
        self._test_connection()
        self._start_background_loop()

        logger.info("BinanceClient initialized - FUTURES MAINNET")

    def _initialize_session(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
        self.session.mount('https://', adapter)
        logger.info("HTTP session initialized with connection pooling")

    def _start_background_loop(self):
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        logger.info("Background event loop started")

    def _check_rate_limit(self) -> bool:
        with self.rate_limit_lock:
            now = time.time()
            if now - self.rate_limit.minute_start >= 60:
                self.rate_limit.minute_start = now
                self.rate_limit.minute_count = 0
            if now - self.rate_limit.last_request_time < 1.0 / self.rate_limit.requests_per_second:
                time.sleep((1.0 / self.rate_limit.requests_per_second) - (now - self.rate_limit.last_request_time))
            if self.rate_limit.minute_count >= self.rate_limit.requests_per_minute:
                sleep = 60 - (now - self.rate_limit.minute_start)
                if sleep > 0:
                    logger.warning(f"Rate limit: sleeping {sleep:.1f}s")
                    time.sleep(sleep)
                    self.rate_limit.minute_start = time.time()
                    self.rate_limit.minute_count = 0
            self.rate_limit.last_request_time = time.time()
            self.rate_limit.minute_count += 1
            return True

    def _check_production_rate_limit(self):
        now = datetime.now(timezone.utc)
        if (now - self._last_reset).total_seconds() >= self.production_rate_limit_interval:
            self._request_count = 0
            self._last_reset = now
        if self._request_count >= self.production_rate_limit_count:
            sleep = self.production_rate_limit_interval - (now - self._last_reset).total_seconds()
            if sleep > 0:
                logger.warning(f"Production limit: sleeping {sleep:.1f}s")
                time.sleep(sleep)
        self._request_count += 1

    def _make_request(self, method: str, endpoint: str, params=None, *, signed: bool = False) -> Optional[Dict]:
        self.total_requests += 1
        self._check_rate_limit()
        self._check_production_rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}

        if signed:
            params = params or {}
            ts = int((time.time() * 1000) + getattr(self.exchange, 'timeDifference', 0))
            params['timestamp'] = str(ts)
            query = '&'.join(f"{k}={v}" for k, v in sorted(params.items()))
            params['signature'] = hmac.new(
                self.api_secret.encode(), query.encode(), hashlib.sha256
            ).hexdigest()

        try:
            resp = self.session.request(
                method, url, params=params, headers=headers,
                timeout=(10, 30)
            )
            resp.raise_for_status()
            data = resp.json()
            self.successful_requests += 1
            self.last_successful_request = datetime.now(timezone.utc)
            return data
        except requests.exceptions.HTTPError as e:
            self.failed_requests += 1
            if e.response.status_code == 429:
                raise APIRateLimitException("Rate limit", endpoint=endpoint)
            elif e.response.status_code in (401, 403):
                raise APIAuthenticationException("Auth failed", endpoint=endpoint)
            raise
        except Exception as e:
            self.failed_requests += 1
            raise APIConnectionException(f"Request failed: {e}", endpoint=endpoint)

    def _test_connection(self):
        try:
            self._make_request("GET", "/fapi/v1/ping", signed=False)
            self._connected = True
            logger.info("Futures connection test: OK")
        except Exception as e:
            self._connected = False
            raise APIConnectionException(f"Connection test failed: {e}")

    def is_connected(self) -> bool:
        if not self._connected:
            try:
                self._test_connection()
            except:
                self._connected = False
        return self._connected

    def get_current_price(self, symbol: str) -> float:
        try:
            if not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '/USDT')
            if TRADING_MODE == "virtual":
                base = {'BTC/USDT': 65000, 'ETH/USDT': 3200}.get(symbol, 100)
                return round(base * random.uniform(0.98, 1.02), 6)
            cached = self._price_cache.get(symbol)
            if cached and (time.time() - cached[0]) < 60:
                return cached[1]
            ticker = self._make_request("GET", "/fapi/v1/ticker/price", {"symbol": symbol.replace('/', '')}, signed=False)
            price = float(ticker["price"])
            self._price_cache[symbol] = (time.time(), price)
            return price
        except Exception as e:
            logger.error(f"Price error {symbol}: {e}")
            return 0.0

    def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            if '/' in symbol:
                symbol = symbol.replace('/', '')
            klines = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [{
                "timestamp": k[0], "open": k[1], "high": k[2],
                "low": k[3], "close": k[4], "volume": k[5]
            } for k in klines]
        except Exception as e:
            logger.error(f"Kline error {symbol}: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Dict:
        try:
            markets = self.exchange.load_markets()
            info = markets.get(symbol, {})
            filters = info.get("info", {}).get("filters", [])
            lot = next((f for f in filters if f["filterType"] == "LOT_SIZE"), {})
            return {
                "lotSizeFilter": {
                    "minOrderQty": float(lot.get("minQty", 0)),
                    "qtyStep": float(lot.get("stepSize", 0.001))
                }
            }
        except Exception as e:
            logger.error(f"Symbol info error {symbol}: {e}")
            return {}

    def get_balance(self) -> Dict[str, float]:
        try:
            data = self._make_request("GET", "/fapi/v2/balance", signed=True)
            usdt = next((a for a in data if a['asset'] == 'USDT'), None)
            if usdt:
                return {
                    'available': float(usdt['availableBalance']),
                    'total': float(usdt['balance'])
                }
            return {'available': 0.0, 'total': 0.0}
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return {'available': 0.0, 'total': 0.0}

    async def place_order(
        self, symbol: str, side: str, qty: float,
        leverage: Optional[int] = None, stopLoss: Optional[float] = None,
        takeProfit: Optional[float] = None, indicators=None
    ) -> Dict:
        db = DatabaseManager()
        try:
            side = side.lower()
            if side not in ('buy', 'sell'):
                raise ValueError("side must be 'buy' or 'sell'")

            if TRADING_MODE == "virtual":
                price = self.get_current_price(symbol)
                order_id = f"virtual_{int(time.time()*1000)}"
                trade = Trade(
                    order_id=order_id, symbol=symbol.replace('/', ''), exchange="binance",
                    virtual=True, side=side, price=price, qty=qty, leverage=leverage or 1,
                    sl=stopLoss, tp=takeProfit, indicators=indicators or {}, status="open",
                    created_at=datetime.now(timezone.utc)
                )
                db.add_trade(trade)
                db.session.commit()
                logger.info(f"Virtual order: {order_id}")
                return {"order_id": order_id, "status": "filled", "virtual": True}

            if leverage:
                await asyncio.to_thread(self.set_leverage, symbol, leverage)

            params = {
                "symbol": symbol.replace('/', ''),
                "side": side.upper(),
                "type": "MARKET",
                "quantity": f"{qty:.6f}".rstrip('0').rstrip('.'),
            }
            result = await asyncio.to_thread(
                self._make_request, "POST", "/fapi/v1/order", params, signed=True
            )
            order_id = result["orderId"]
            trade = Trade(
                order_id=str(order_id), symbol=symbol.replace('/', ''), exchange="binance",
                virtual=False, side=side, price=self.get_current_price(symbol), qty=qty,
                leverage=leverage or 1, sl=stopLoss, tp=takeProfit,
                indicators=indicators or {}, status="open",
                created_at=datetime.now(timezone.utc)
            )
            db.add_trade(trade)
            db.session.commit()
            logger.info(f"Order placed: {order_id}")
            return {"order_id": order_id, "status": "filled"}
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {"error": str(e)}
        finally:
            db.session.close()

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            self._make_request("POST", "/fapi/v1/leverage", {
                "symbol": symbol.replace('/', ''), "leverage": str(leverage)
            }, signed=True)
            logger.info(f"Leverage {leverage}x set")
            return True
        except Exception as e:
            logger.error(f"Leverage error: {e}")
            return False

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            params = {"symbol": symbol.replace('/', '')} if symbol else {}
            data = await asyncio.to_thread(
                self._make_request, "GET", "/fapi/v2/positionRisk", params, signed=True
            )
            positions = []
            for p in data:
                amt = float(p["positionAmt"])
                if amt != 0:
                    positions.append({
                        "symbol": p["symbol"],
                        "side": "buy" if amt > 0 else "sell",
                        "size": abs(amt),
                        "price": float(p["entryPrice"]),
                        "unrealized_pnl": float(p["unRealizedProfit"]),
                        "leverage": float(p["leverage"]),
                    })
            return positions
        except Exception as e:
            logger.error(f"Positions error: {e}")
            return []

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        db = DatabaseManager()
        try:
            if TRADING_MODE == "virtual" and order_id.startswith("virtual_"):
                trade = db.get_trade(order_id)
                if trade:
                    db.update_trade(cast(int, trade.id), {"status": "canceled"})
                return True
            result = await asyncio.to_thread(
                self._make_request, "DELETE", "/fapi/v1/order",
                {"symbol": symbol.replace('/', ''), "orderId": order_id}, signed=True
            )
            if result:
                trade = db.get_trade(order_id)
                if trade:
                    db.update_trade(cast(int, trade.id), {"status": "canceled"})
                return True
            return False
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False
        finally:
            db.session.close()

    async def start_websocket(self, symbols: List[str]):
        if not self.loop:
            return
        async def handler():
            stream = '@'.join([f'{s.lower().replace("/", "")}@ticker' for s in symbols])
            uri = f"{self.ws_url}/ws/{stream}"
            try:
                async with websockets.connect(uri) as ws:
                    self.ws_connection = ws
                    async for msg in ws:
                        data = json.loads(msg)
                        sym = data.get("s")
                        price = float(data.get("c", 0))
                        if sym and price > 0:
                            self._price_cache[sym] = (time.time(), price)
            except Exception as e:
                logger.error(f"WS error: {e}")
        asyncio.run_coroutine_threadsafe(handler(), self.loop)

    def close(self):
        try:
            if self.ws_connection and self.loop:
                asyncio.run_coroutine_threadsafe(self.ws_connection.close(), self.loop)
            if self.session:
                self.session.close()
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            logger.info("Binance client closed")
        except Exception as e:
            logger.error(f"Close error: {e}")