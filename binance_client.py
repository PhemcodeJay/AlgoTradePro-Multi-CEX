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

if TYPE_CHECKING:
    from db import DatabaseManager, Trade, WalletBalance
else:
    from db import DatabaseManager, Trade, WalletBalance

logger = get_trading_logger('api_client')

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

@dataclass
class RateLimitInfo:
    requests_per_second: int = 10
    requests_per_minute: int = 600
    last_request_time: float = 0
    minute_start: float = 0
    minute_count: int = 0

class BinanceClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        try:
            self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
            self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
            if not self.api_key or not self.api_secret:
                raise APIConnectionException("Binance API key or secret not provided")
        except Exception as e:
            raise APIConnectionException(f"Failed to initialize API credentials: {str(e)}")

        self.account_type = os.getenv("BINANCE_ACCOUNT_TYPE", "FUTURES").upper()
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443"
        self.session = None
        self.ws_connection = None
        self.loop = None
        self._price_cache: Dict[str, tuple[float, float]] = {}
        self._connected = False
        self._connection_lock = threading.Lock()

        self.rate_limit = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        self.production_rate_limit_interval = 1
        self.production_rate_limit_count = 10

        self.recovery_strategy = APIErrorRecoveryStrategy(max_retries=3, delay=1.0)
        self.last_error_time = None
        self.consecutive_errors = 0

        self.request_timeout = 30
        self.connect_timeout = 10

        self.last_successful_request = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'timeout': 30000,
        })
        self.exchange.set_sandbox_mode(False)

        self._initialize_session()
        self._test_connection()
        self._start_background_loop()

        logger.info(f"BinanceClient initialized - mainnet - {self.account_type}")

    def _initialize_session(self):
        try:
            self.session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=0
            )
            self.session.mount('https://', adapter)
            self.session.mount('http://', adapter)
            logger.info("HTTP session initialized with connection pooling")
        except Exception as e:
            error_context = create_error_context(module=__name__, function='_initialize_session')
            raise APIConnectionException(
                f"Failed to initialize HTTP session: {str(e)}",
                endpoint='session_init',
                context=error_context,
                original_exception=e
            )

    def _start_background_loop(self):
        try:
            def run_loop():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()
            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()
            logger.info("Background event loop started")
        except Exception as e:
            logger.error(f"Failed to start background loop: {e}")

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }

    def _check_rate_limit(self) -> bool:
        with self.rate_limit_lock:
            now = time.time()
            if now - self.rate_limit.minute_start >= 60:
                self.rate_limit.minute_start = now
                self.rate_limit.minute_count = 0
            time_since_last = now - self.rate_limit.last_request_time
            if time_since_last < 1.0 / self.rate_limit.requests_per_second:
                sleep_time = (1.0 / self.rate_limit.requests_per_second) - time_since_last
                time.sleep(sleep_time)
                now = time.time()
            if self.rate_limit.minute_count >= self.rate_limit.requests_per_minute:
                sleep_time = 60 - (now - self.rate_limit.minute_start)
                if sleep_time > 0:
                    logger.warning(f"Rate limit hit, sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                    self.rate_limit.minute_start = time.time()
                    self.rate_limit.minute_count = 0
            self.rate_limit.last_request_time = time.time()
            self.rate_limit.minute_count += 1
            return True

    def _check_production_rate_limit(self) -> None:
        now = datetime.now(timezone.utc)
        if (now - self._last_reset).total_seconds() >= self.production_rate_limit_interval:
            self._request_count = 0
            self._last_reset = now
        if self._request_count >= self.production_rate_limit_count:
            sleep_time = self.production_rate_limit_interval - (now - self._last_reset).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Production rate limit hit, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        self._request_count += 1

    def _validate_api_credentials(self) -> bool:
        if not self.api_key or not self.api_secret:
            raise APIAuthenticationException(
                "API credentials not configured. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.",
                context=create_error_context(module=__name__, function='_validate_api_credentials')
            )
        return True

    def _handle_api_error(self, response_data: Dict, endpoint: str) -> None:
        error_code = response_data.get("code", -1)
        error_msg = response_data.get("msg", "Unknown error")
        if error_code == -1003:
            raise APIRateLimitException(f"Rate limit exceeded: {error_msg}", retry_after=60, context=create_error_context(module=__name__, function='_handle_api_error'))
        elif error_code in (-2014, -2015):
            raise APIAuthenticationException(f"Authentication failed: {error_msg}", context=create_error_context(module=__name__, function='_handle_api_error'))
        else:
            raise APIException(f"API Error (code {error_code}): {error_msg}", error_code=str(error_code), context=create_error_context(module=__name__, function='_handle_api_error', extra_data={'endpoint': endpoint, 'error_code': error_code}))

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        *,
        signed: bool = False
    ) -> Optional[Dict]:
        try:
            self.total_requests += 1
            self._check_rate_limit()

            url = f"{self.base_url}{endpoint}"
            headers = {'X-MBX-APIKEY': self.api_key}

            # Only add timestamp + signature for signed (private) endpoints
            if signed:
                if params is None:
                    params = {}
                params['timestamp'] = str(int(time.time() * 1000))
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                params['signature'] = hmac.new(
                    self.api_secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()

            response = self.session.request(
                method,
                url,
                params=params if params else None,
                headers=headers,
                timeout=(self.connect_timeout, self.request_timeout)
            )
            response.raise_for_status()
            result = response.json()
            self.successful_requests += 1
            self.last_successful_request = datetime.now(timezone.utc)
            self.consecutive_errors = 0
            return result

        except requests.exceptions.RequestException as e:
            self.failed_requests += 1
            logger.error(f"API request failed for {endpoint}: {str(e)}", exc_info=True)
            if isinstance(e, requests.exceptions.HTTPError):
                if e.response.status_code == 429:
                    raise APIRateLimitException("Rate limit exceeded", endpoint=endpoint)
                elif e.response.status_code in [401, 403]:
                    raise APIAuthenticationException("Authentication error", endpoint=endpoint)
                elif e.response.status_code == 504:
                    raise APITimeoutException("Request timeout", endpoint=endpoint)
            raise APIConnectionException(f"Request failed: {str(e)}", endpoint=endpoint, original_exception=e)

    def _test_connection(self):
        try:
            # /ping is public → no signature
            self._make_request("GET", "/api/v3/ping", signed=False)
            self._connected = True
            logger.info("Binance connection test successful")
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            self._connected = False
            raise APIConnectionException(f"Connection test failed: {str(e)}")

    def get_connection_health(self) -> Dict[str, Any]:
        return {
            'status': 'healthy' if self._connected else 'unhealthy',
            'last_successful_request': self.last_successful_request,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests
        }

    def is_connected(self) -> bool:
        if not self._connected:
            try:
                self._test_connection()
            except Exception as e:
                logger.warning(f"Connection verification failed: {e}")
                self._connected = False
        return self._connected and bool(self.api_key and self.api_secret)

    def get_current_price(self, symbol: str) -> float:
        try:
            if not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '/USDT')
            if TRADING_MODE == "virtual":
                base_prices = {
                    'BTC/USDT': 65000, 'ETH/USDT': 3200, 'DOGE/USDT': 0.15,
                    'SOL/USDT': 150, 'XRP/USDT': 0.55, 'BNB/USDT': 600, 'AVAX/USDT': 35,
                    'LTC/USDT': 70, 'MATIC/USDT': 0.8, 'SHIB/USDT': 0.00002,
                    'TRX/USDT': 0.15, 'VET/USDT': 0.03, 'XLM/USDT': 0.1
                }
                price = base_prices.get(symbol, 100) * random.uniform(0.98, 1.02)
                return round(price, 6)
            cached = self._price_cache.get(symbol)
            if cached and (time.time() - cached[0]) < 60:
                return cached[1]
            ticker = self._make_request("GET", "/api/v3/ticker/price", {"symbol": symbol.replace('/', '')}, signed=False)
            price = float(ticker.get("price", 0))
            if price > 0:
                self._price_cache[symbol] = (time.time(), price)
            return price
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_current_over_price(self, symbol: str) -> float:
        return self.get_current_price(symbol)

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        try:
            if TRADING_MODE == "virtual":
                logger.info(f"Virtual mode: Generating simulated klines for {symbol}")
                return self._generate_simulated_klines(symbol, limit)
            
            timeframe_map = {'1': '1m', '5': '5m', '15': '15m', '30': '30m', '60': '1h', '240': '4h', '1440': '1d'}
            timeframe = timeframe_map.get(interval, '1h')
            url = f"{self.base_url}/api/v3/klines"
            params = {
                "symbol": symbol.replace('/', ''),
                "interval": timeframe,
                "limit": str(limit)
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, dict):
                code = result.get('code', None)
                msg = result.get('msg', '')
                if code == 0 and 'restricted location' in msg.lower():
                    raise APIException(f"Binance API restricted for {symbol}: {msg}. Cannot fetch real data in real mode.")
                else:
                    raise APIDataException(f"Unexpected API response for {symbol}: {result}")
            
            klines = []
            for k in result:
                klines.append({
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })
            logger.info(f"Successfully fetched {len(klines)} real klines for {symbol}")
            return sorted(klines, key=lambda x: x["timestamp"])
        except APIException as e:
            logger.error(f"API restriction error for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting real klines for {symbol}: {e}")
            raise APIConnectionException(f"Failed to fetch klines for {symbol}: {str(e)}")

    def _generate_simulated_klines(self, symbol: str, limit: int) -> List[Dict]:
        base_prices = {
            'BTCUSDT': 65000, 'ETHUSDT': 3200, 'DOGEUSDT': 0.15,
            'SOLUSDT': 150, 'XRPUSDT': 0.55, 'BNBUSDT': 600, 'AVAXUSDT': 35,
            'LTCUSDT': 70, 'MATICUSDT': 0.8, 'SHIBUSDT': 0.00002,
            'TRXUSDT': 0.15, 'VETUSDT': 0.03, 'XLMUSDT': 0.1
        }
        base_price = base_prices.get(symbol, 100)
        klines = []
        current_time = int(time.time() * 1000)
        interval_ms = 3600000  # 1 hour
        limit = max(limit, 200)
        for i in range(limit):
            timestamp = current_time - (limit - i) * interval_ms
            open_price = base_price * (1 + random.uniform(-0.05, 0.05))
            close_price = open_price * (1 + random.uniform(-0.07, 0.07))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.03))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.03))
            volume = random.uniform(1000000, 5000000)
            klines.append({
                'timestamp': timestamp,
                'open': round(open_price, 6),
                'high': round(high_price, 6),
                'low': round(low_price, 6),
                'close': round(close_price, 6),
                'volume': round(volume, 2)
            })
            base_price = close_price
        logger.debug(f"Generated {len(klines)} simulated klines for {symbol}")
        return klines
    
    def get_balance(self) -> Dict[str, float]:
        try:
            result = self._make_request("GET", "/fapi/v2/balance", signed=True)
            for asset in result:
                if asset['asset'] == 'USDT':
                    return {
                        'available': float(asset['availableBalance']),
                        'total': float(asset['balance'])
                    }
            return {'available': 0.0, 'total': 0.0}
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {'available': 0.0, 'total': 0.0}

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        leverage: Optional[int] = None,
        mode: str = "CROSS",
        stopLoss: Optional[float] = None,
        takeProfit: Optional[float] = None,
        indicators: Optional[Dict[str, Any]] = None
    ) -> Dict:
        db_manager = DatabaseManager()
        try:
            side = side.lower()
            if side not in ('buy', 'sell'):
                raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
            if not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '/USDT')
            if TRADING_MODE == "virtual":
                price = self.get_current_price(symbol)
                if price <= 0:
                    raise ValueError(f"Invalid price for {symbol}: {price}")
                order_id = f"virtual_{int(time.time() * 1000)}"
                order = {
                    "order_id": order_id,
                    "symbol": symbol,
                    "accountType": self.account_type,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "status": "filled",
                    "timestamp": datetime.now(),
                    "virtual": True,
                    "leverage": leverage or 1,
                    "stopLoss": str(stopLoss) if stopLoss is not None else None,
                    "takeProfit": str(takeProfit) if takeProfit is not None else None,
                    "margin_mode": mode.upper()
                }
                trade = Trade(
                    order_id=order_id,
                    symbol=symbol.replace('/', ''),
                    exchange="binance",
                    virtual=True,
                    side=side,
                    price=price,
                    qty=qty,
                    leverage=leverage or 1,
                    sl=stopLoss,
                    tp=takeProfit,
                    indicators=indicators or {},
                    status="open",
                    created_at=datetime.now(timezone.utc)
                )
                db_manager.add_trade(trade)
                db_manager.session.commit()
                logger.info(f"Virtual order placed: {order_id} - {symbol} {side} {qty}")
                return order
            if leverage and self.account_type == "FUTURES":
                await asyncio.to_thread(self.set_leverage, symbol, leverage)
            params = {
                "symbol": symbol.replace('/', ''),
                "side": side.upper(),
                "quantity": str(qty),
                "type": "MARKET",
                "timeInForce": "GTC"
            }
            if stopLoss is not None:
                params["stopLossPrice"] = str(round(float(stopLoss), 6))
            if takeProfit is not None:
                params["takeProfitPrice"] = str(round(float(takeProfit), 6))
            result = await asyncio.to_thread(self._make_request, "POST", "/api/v3/order", params, signed=True)
            if result:
                order_id = result.get("orderId")
                order = {
                    "order_id": str(order_id),
                    "symbol": symbol,
                    "accountType": self.account_type,
                    "side": side,
                    "qty": qty,
                    "price": float(result.get("price", 0) or self.get_current_price(symbol)),
                    "status": result.get("status", "pending").lower(),
                    "timestamp": datetime.fromtimestamp(int(result.get("time", 0)) / 1000),
                    "virtual": False,
                    "leverage": leverage or 1,
                    "stopLoss": str(stopLoss) if stopLoss is not None else None,
                    "takeProfit": str(takeProfit) if takeProfit is not None else None,
                    "margin_mode": mode.upper()
                }
                trade = Trade(
                    order_id=str(order_id),
                    symbol=symbol.replace('/', ''),
                    exchange="binance",
                    virtual=False,
                    side=side,
                    price=order["price"],
                    qty=qty,
                    leverage=leverage or 1,
                    sl=stopLoss,
                    tp=takeProfit,
                    indicators=indicators or {},
                    status="open",
                    created_at=datetime.now(timezone.utc)
                )
                db_manager.add_trade(trade)
                db_manager.session.commit()
                logger.info(f"Order placed: {order_id} - {symbol} {side} {qty}")
                return order
            return {}
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {"error": str(e)}
        finally:
            db_manager.session.close()

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        db_manager = DatabaseManager()
        try:
            if not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '/USDT')
            if TRADING_MODE == "virtual" and order_id.startswith("virtual_"):
                if hasattr(db_manager, 'get_trade'):
                    trade = db_manager.get_trade(order_id)
                    if trade:
                        db_manager.update_trade(cast(int, trade.id), {"status": "canceled"})
                        logger.info(f"Virtual order canceled: {order_id} - {symbol}")
                        return True
                    return False
                else:
                    logger.warning("DatabaseManager.get_trade not available, assuming virtual order cancellation")
                    return True
            params = {"symbol": symbol.replace('/', ''), "orderId": order_id}
            result = await asyncio.to_thread(self._make_request, "DELETE", "/api/v3/order", params, signed=True)
            if result:
                if hasattr(db_manager, 'get_trade'):
                    trade = db_manager.get_trade(order_id)
                    if trade:
                        db_manager.update_trade(cast(int, trade.id), {"status": "canceled"})
                logger.info(f"Order canceled: {order_id} - {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
        finally:
            db_manager.session.close()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            if symbol and not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '/USDT')
            params = {"symbol": symbol.replace('/', '')} if symbol else {}
            result = self._make_request("GET", "/api/v3/openOrders", params, signed=True)
            orders: List[Dict[str, Any]] = []
            for order in result:
                orders.append({
                    "order_id": str(order.get("orderId")),
                    "symbol": order.get("symbol"),
                    "side": order.get("side").lower(),
                    "qty": float(order.get("origQty", 0)),
                    "price": float(order.get("price", 0)),
                    "status": order.get("status").lower(),
                    "timestamp": datetime.fromtimestamp(int(order.get("time", 0)) / 1000),
                    "virtual": False
                })
            return orders
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        db_manager = DatabaseManager()
        try:
            if symbol and not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('/', '')
            params = {"symbol": symbol} if symbol else {}
            result = await asyncio.to_thread(self._make_request, "GET", "/fapi/v2/positionRisk", params, signed=True)
            positions: List[Dict] = []
            for pos in result:
                size = float(pos.get("positionAmt", 0))
                if size != 0:
                    position = {
                        "symbol": pos.get("symbol"),
                        "side": "buy" if float(pos.get("positionAmt", 0)) > 0 else "sell",
                        "size": abs(size),
                        "price": float(pos.get("entryPrice", 0)),
                        "mark_price": float(pos.get("markPrice", 0)),
                        "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                        "leverage": float(pos.get("leverage", 1)),
                        "virtual": False
                    }
                    if hasattr(db_manager, 'get_trade_by_position'):
                        trade = db_manager.get_trade_by_position(pos.get("symbol"), "binance", False)
                        if trade:
                            db_manager.update_trade(cast(int, trade.id), {
                                "pnl": position["unrealized_pnl"],
                                "status": "open"
                            })
                    positions.append(position)
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
        finally:
            db_manager.session.close()

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            if not '/' in symbol and symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '/USDT')
            params = {"symbol": symbol.replace('/', ''), "leverage": str(leverage)}
            self._make_request("POST", "/fapi/v1/leverage", params, signed=True)
            logger.info(f"Binance leverage set to {leverage}x for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error setting Binance leverage: {e}")
            return False

    def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        try:
            result = self._make_request("GET", "/api/v3/account", signed=True)
            fees: Dict[str, Any] = {}
            if result and "makerCommission" in result and "takerCommission" in result:
                fee_rate = {
                    "maker_fee": float(result["makerCommission"]) / 10000,
                    "taker_fee": float(result["takerCommission"]) / 10000
                }
                if symbol:
                    fees[symbol.replace('/', '')] = fee_rate
                else:
                    fees["default"] = fee_rate
            return fees
        except Exception as e:
            logger.error(f"Error fetching Binance trading fees: {e}")
            return {}

    async def start_websocket(self, symbols: List[str]):
        try:
            if not self.loop:
                logger.error("Event loop not available")
                return
            async def websocket_handler():
                stream_names = '@'.join([f'{s.lower().replace("/", "")}@ticker' for s in symbols])
                uri = f"{self.ws_url}/ws/{stream_names}"
                try:
                    async with websockets.connect(uri) as websocket:
                        self.ws_connection = websocket
                        async for message in websocket:
                            data = json.loads(message)
                            symbol = data.get("s")
                            price = float(data.get("c", 0))
                            if symbol and price > 0:
                                self._price_cache[symbol] = (time.time(), price)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    self.ws_connection = None
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(websocket_handler(), self.loop)
                logger.info("WebSocket connection started")
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")

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
            logger.error(f"Error closing client: {e}")