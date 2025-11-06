import os
import hmac
import hashlib
import time
import json
import asyncio
import websockets
from typing import Dict, Any, Optional, List, TYPE_CHECKING, cast
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
import threading
from dataclasses import dataclass
from logging_config import get_logger
from exceptions import (
    APIException, APIConnectionException, APIRateLimitException,
    APIAuthenticationException, APITimeoutException, APIDataException,
    APIErrorRecoveryStrategy, create_error_context
)
from dotenv import load_dotenv  # <-- ADDED

load_dotenv()

if TYPE_CHECKING:
    from db import DatabaseManager, Trade, WalletBalance
else:
    from db import DatabaseManager, Trade, WalletBalance

logger = get_logger('bybit_api_client')
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()


@dataclass
class RateLimitInfo:
    requests_per_second: int = 10
    requests_per_minute: int = 300
    last_request_time: float = 0.0
    minute_start: float = 0.0
    minute_count: int = 0


class BybitClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        if not self.api_key or not self.api_secret:
            raise APIConnectionException("BYBIT_API_KEY and BYBIT_API_SECRET required in .env")

        self.account_type = os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED").upper()
        self.base_url = "https://api.bybit.com"
        self.ws_url = "wss://stream.bybit.com"
        self.session = None
        self.ws_connection = None
        self.loop = None
        self._price_cache: Dict[str, tuple[float, float]] = {}
        self._connected = False

        self.rate_limit = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.last_successful_request = None

        self._initialize_session()
        self._test_connection()
        self._start_background_loop()

        logger.info(f"BybitClient initialized - mainnet - {self.account_type}")

    def _initialize_session(self):
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
        self.session.mount('https://', adapter)
        logger.info("HTTP session ready")

    def _start_background_loop(self):
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        threading.Thread(target=run_loop, daemon=True).start()
        logger.info("WebSocket loop started")

    def _generate_signature(self, method: str, endpoint: str, params: Dict) -> str:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        param_str = timestamp + self.api_key + recv_window + (json.dumps(params) if method == "POST" else "")
        return hmac.new(self.api_secret.encode(), param_str.encode(), hashlib.sha256).hexdigest()

    def _get_headers(self, method: str, endpoint: str, params: Dict) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(method, endpoint, params)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }

    def _check_rate_limit(self):
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
                    logger.warning(f"Rate limit → sleep {sleep:.1f}s")
                    time.sleep(sleep)
                    self.rate_limit.minute_start = time.time()
                    self.rate_limit.minute_count = 0
            self.rate_limit.last_request_time = time.time()
            self.rate_limit.minute_count += 1

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        self.total_requests += 1
        self._check_rate_limit()
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        headers = self._get_headers(method, endpoint, params)
        try:
            if method == "GET":
                resp = self.session.get(url, params=params, headers=headers, timeout=(10, 30))
            else:
                resp = self.session.post(url, json=params, headers=headers, timeout=(10, 30))
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode") != 0:
                self._handle_error(data, endpoint)
            self.successful_requests += 1
            self.last_successful_request = datetime.now(timezone.utc)
            return data.get("result", {})
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

    def _handle_error(self, data: Dict, endpoint: str):
        code = data.get("retCode", -1)
        msg = data.get("retMsg", "Unknown")
        if code == 10003:
            raise APIRateLimitException(f"Rate limit: {msg}", retry_after=60)
        elif code in (10002, 10004):
            raise APIAuthenticationException(f"Auth error: {msg}")
        elif code == 100028:
            raise APIException(f"Forbidden (unified account): {msg}. Use CROSS mode.")
        else:
            raise APIException(f"API error {code}: {msg}", error_code=str(code))

    def _test_connection(self):
        try:
            self._make_request("GET", "/v5/market/time", {})
            self._connected = True
            logger.info("Bybit ping OK")
        except:
            self._connected = False
            raise APIConnectionException("Connection failed")

    def is_connected(self) -> bool:
        if not self._connected:
            try:
                self._test_connection()
            except:
                return False
        return True

    def get_current_price(self, symbol: str) -> float:
        try:
            cached = self._price_cache.get(symbol)
            if cached and (time.time() - cached[0]) < 10:
                return cached[1]
            resp = requests.get(f"{self.base_url}/v5/market/tickers", params={"category": "linear", "symbol": symbol}, timeout=10)
            resp.raise_for_status()
            price = float(resp.json()["result"]["list"][0]["lastPrice"])
            self._price_cache[symbol] = (time.time(), price)
            return price
        except:
            return 0.0

    def get_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            result = self._make_request("GET", "/v5/market/kline", {
                "category": "linear", "symbol": symbol, "interval": timeframe, "limit": limit
            })
            klines = []
            for k in result.get("list", []):
                klines.append({
                    "timestamp": int(k[0]), "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])
                })
            return klines[::-1]
        except:
            return []

    def get_symbol_info(self, symbol: str) -> Dict:
        try:
            result = self._make_request("GET", "/v5/market/instruments-info", {
                "category": "linear", "symbol": symbol
            })
            info = result["list"][0]
            lot = info.get("lotSizeFilter", {})
            return {
                "lotSizeFilter": {
                    "minOrderQty": float(lot.get("minOrderQty", 0)),
                    "qtyStep": float(lot.get("qtyStep", 0.001))
                }
            }
        except:
            return {}

    def get_balance(self) -> Dict[str, float]:
        try:
            result = self._make_request("GET", "/v5/account/wallet-balance", {"accountType": self.account_type})
            usdt = next((c for c in result["list"][0]["coin"] if c["coin"] == "USDT"), None)
            if usdt:
                return {
                    "available": float(usdt["availableToWithdraw"]),
                    "total": float(usdt["walletBalance"])
                }
            return {"available": 0.0, "total": 0.0}
        except:
            return {"available": 0.0, "total": 0.0}

    async def place_order(
        self, symbol: str, side: str, qty: float,
        leverage: int = 10, stopLoss: Optional[float] = None,
        takeProfit: Optional[float] = None, indicators=None
    ) -> Dict:
        db = DatabaseManager()
        try:
            side = side.lower()
            if TRADING_MODE == "virtual":
                price = self.get_current_price(symbol)
                order_id = f"virtual_{int(time.time()*1000)}"
                trade = Trade(
                    order_id=order_id, symbol=symbol, exchange="bybit", virtual=True,
                    side=side, entry_price=price, qty=qty, leverage=leverage,
                    sl=stopLoss, tp=takeProfit, indicators=indicators or {}, status="open",
                    created_at=datetime.now(timezone.utc)
                )
                db.add_trade(trade)
                db.session.commit()
                return {"order_id": order_id, "status": "filled", "virtual": True}

            # Unified account → force CROSS
            if self.account_type == "UNIFIED":
                leverage = 1  # Not used in unified

            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.title(),
                "orderType": "Market",
                "qty": f"{qty:.6f}".rstrip('0').rstrip('.'),
                "timeInForce": "IOC"
            }
            if stopLoss: params["stopLoss"] = str(round(stopLoss, 6))
            if takeProfit: params["takeProfit"] = str(round(takeProfit, 6))

            result = await asyncio.to_thread(self._make_request, "POST", "/v5/order/create", params)
            order_id = result["orderId"]
            trade = Trade(
                order_id=order_id, symbol=symbol, exchange="bybit", virtual=False,
                side=side, entry_price=self.get_current_price(symbol), qty=qty,
                leverage=leverage, sl=stopLoss, tp=takeProfit,
                indicators=indicators or {}, status="open",
                created_at=datetime.now(timezone.utc)
            )
            db.add_trade(trade)
            db.session.commit()
            return {"order_id": order_id, "status": "filled"}
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {"error": str(e)}
        finally:
            db.session.close()

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        db = DatabaseManager()
        try:
            if TRADING_MODE == "virtual" and order_id.startswith("virtual_"):
                trade = db.get_trade(order_id)
                if trade:
                    db.update_trade(cast(int, trade.id), {"status": "canceled"})
                return True
            await asyncio.to_thread(self._make_request, "POST", "/v5/order/cancel", {
                "category": "linear", "symbol": symbol, "orderId": order_id
            })
            trade = db.get_trade(order_id)
            if trade:
                db.update_trade(cast(int, trade.id), {"status": "canceled"})
            return True
        except:
            return False
        finally:
            db.session.close()

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            params = {"category": "linear"}
            if symbol: params["symbol"] = symbol
            result = await asyncio.to_thread(self._make_request, "GET", "/v5/position/list", params)
            positions = []
            for p in result.get("list", []):
                size = float(p.get("size", 0))
                if size > 0:
                    positions.append({
                        "symbol": p["symbol"],
                        "side": p["side"].lower(),
                        "size": size,
                        "entry_price": float(p["avgPrice"]),
                        "unrealized_pnl": float(p["unrealisedPnl"]),
                        "leverage": float(p.get("leverage", 10))
                    })
            return positions
        except:
            return []

    async def start_websocket(self, symbols: List[str]):
        if not self.loop:
            return
        async def handler():
            uri = f"{self.ws_url}/v5/public/linear"
            try:
                async with websockets.connect(uri) as ws:
                    self.ws_connection = ws
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": [f"tickers.{s}" for s in symbols]
                    }))
                    async for msg in ws:
                        data = json.loads(msg)
                        if data.get("topic", "").startswith("tickers."):
                            d = data["data"]
                            sym = d["symbol"]
                            price = float(d["lastPrice"])
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
            logger.info("Bybit client closed")
        except:
            pass