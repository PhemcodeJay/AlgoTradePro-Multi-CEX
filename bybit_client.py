import os
import hmac
import hashlib
import time
import json
import asyncio
import websockets
from typing import Dict, Any, Optional, List, TYPE_CHECKING, cast
import requests.adapters
from datetime import datetime, timezone
import requests
import threading
from dataclasses import dataclass
from logging_config import get_logger
from exceptions import (
    APIException, APIConnectionException, APIRateLimitException, 
    APIAuthenticationException, APITimeoutException, APIDataException,
    APIErrorRecoveryStrategy, create_error_context
)

# Import types for static analysis only
if TYPE_CHECKING:
    from db import DatabaseManager, Trade, Signal, WalletBalance
else:
    from db import DatabaseManager, Trade, Signal

logger = get_logger('api_client')

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

@dataclass
class RateLimitInfo:
    requests_per_second: int = 5
    requests_per_minute: int = 120
    last_request_time: float = 0
    request_count: int = 0
    minute_start: float = 0
    minute_count: int = 0

class BybitClient:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        self.account_type = os.getenv("BYBIT_ACCOUNT_TYPE", "UNIFIED").upper()

        self.base_url = "https://api.bybit.com"
        self.ws_url = "wss://stream.bybit.com"

        if not self.api_key or not self.api_secret:
            raise APIConnectionException("Bybit API key or secret not provided")

        self.session = None
        self.ws_connection = None
        self.loop = None
        self._price_cache: Dict[str, tuple[float, float]] = {}
        self._connected = False
        self._connection_lock = threading.Lock()
        
        self.rate_limit = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        
        self.recovery_strategy = APIErrorRecoveryStrategy(max_retries=3, delay=1.0)
        self.last_error_time = None
        self.consecutive_errors = 0
        
        self.request_timeout = 30
        self.connect_timeout = 10
        
        self.last_successful_request = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        self._initialize_session()
        self._test_connection()
        self._start_background_loop()
        
        logger.info(f"BybitClient initialized - mainnet - {self.account_type}")
    
    def _initialize_session(self):
        try:
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
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

    def _generate_signature(self, params: str, timestamp: str) -> str:
        param_str = timestamp + self.api_key + "5000" + params
        return hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self, params: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(params, timestamp)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
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
    
    def _validate_api_credentials(self) -> bool:
        if not self.api_key or not self.api_secret:
            raise APIAuthenticationException(
                "API credentials not configured. Please set BYBIT_API_KEY and BYBIT_API_SECRET environment variables.",
                context=create_error_context(module=__name__, function='_validate_api_credentials')
            )
        return True
    
    def _handle_api_error(self, response_data: Dict, endpoint: str) -> None:
        ret_code = response_data.get("retCode", -1)
        ret_msg = response_data.get("retMsg", "Unknown error")
        if ret_code == 10001:
            raise APIDataException(f"Request parameter error: {ret_msg}", context=create_error_context(module=__name__, function='_handle_api_error'))
        elif ret_code == 10002:
            raise APIAuthenticationException(f"Request timestamp expired: {ret_msg}", context=create_error_context(module=__name__, function='_handle_api_error'))
        elif ret_code == 10003:
            raise APIRateLimitException(f"Rate limit exceeded: {ret_msg}", retry_after=60, context=create_error_context(module=__name__, function='_handle_api_error'))
        elif ret_code == 10004:
            raise APIAuthenticationException(f"Invalid signature: {ret_msg}", context=create_error_context(module=__name__, function='_handle_api_error'))
        elif ret_code == 100028:
            raise APIException(f"Operation forbidden for unified account: {ret_msg}. Use cross margin mode.", error_code=str(ret_code), context=create_error_context(module=__name__, function='_handle_api_error', extra_data={'endpoint': endpoint, 'ret_code': ret_code}))
        else:
            raise APIException(f"API Error (code {ret_code}): {ret_msg}", error_code=str(ret_code), context=create_error_context(module=__name__, function='_handle_api_error', extra_data={'endpoint': endpoint, 'ret_code': ret_code}))

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        self._validate_api_credentials()
        self._check_rate_limit()
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        start_time = time.time()
        self.total_requests += 1
        for attempt in range(self.recovery_strategy.max_retries + 1):
            try:
                with self._connection_lock:
                    if not self.session:
                        raise APIConnectionException("HTTP session not initialized", endpoint=endpoint)
                    if method.upper() == "GET":
                        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                        headers = self._get_headers(query_string)
                        response = self.session.get(url, params=params, headers=headers, timeout=(self.connect_timeout, self.request_timeout))
                    else:
                        params_str = json.dumps(params) if params else ""
                        headers = self._get_headers(params_str)
                        response = self.session.post(url, json=params, headers=headers, timeout=(self.connect_timeout, self.request_timeout))
                response_time = (time.time() - start_time) * 1000
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise APIRateLimitException(f"Rate limit exceeded on {endpoint}", retry_after=retry_after, context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'attempt': attempt}))
                response.raise_for_status()
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    raise APIDataException(f"Invalid JSON response from {endpoint}: {str(e)}", response_data=response.text[:1000], context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'status_code': response.status_code}))
                if data.get("retCode") != 0:
                    self._handle_api_error(data, endpoint)
                self.successful_requests += 1
                self.last_successful_request = datetime.now()
                self.consecutive_errors = 0
                logger.info(f"API request successful: {method} {endpoint}", extra={'endpoint': endpoint, 'method': method, 'response_time_ms': round(response_time, 2), 'status_code': response.status_code, 'attempt': attempt + 1})
                return data.get("result", {})
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                if not self.recovery_strategy.should_retry(e, attempt):
                    raise APITimeoutException(f"Request timeout for {endpoint} after {attempt + 1} attempts: {str(e)}", timeout_duration=self.request_timeout, context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'total_attempts': attempt + 1}), original_exception=e)
                if attempt < self.recovery_strategy.max_retries:
                    retry_delay = self.recovery_strategy.get_delay(attempt)
                    logger.warning(f"API request failed (attempt {attempt + 1}), retrying in {retry_delay}s: {str(e)}", extra={'endpoint': endpoint, 'attempt': attempt + 1, 'retry_delay': retry_delay})
                    time.sleep(retry_delay)
            except requests.exceptions.HTTPError as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                status_code = e.response.status_code if e.response else None
                if status_code == 401:
                    raise APIAuthenticationException(f"Authentication failed for {endpoint}: {str(e)}", context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'status_code': status_code}), original_exception=e)
                elif status_code == 403:
                    raise APIAuthenticationException(f"Access forbidden for {endpoint}: {str(e)}", context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'status_code': status_code}), original_exception=e)
                else:
                    raise APIConnectionException(f"HTTP error for {endpoint}: {str(e)}", endpoint=endpoint, status_code=status_code, context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'status_code': status_code}), original_exception=e)
            except Exception as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                logger.error(f"Unexpected error in API request to {endpoint}: {str(e)}", extra={'endpoint': endpoint, 'attempt': attempt + 1})
                raise APIException(f"Unexpected error for {endpoint}: {str(e)}", context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'attempt': attempt + 1}), original_exception=e)
        raise APIException(f"Maximum retry attempts exceeded for {endpoint}", context=create_error_context(module=__name__, function='_make_request', extra_data={'endpoint': endpoint, 'max_retries': self.recovery_strategy.max_retries}))

    async def _make_request_async(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        return await asyncio.to_thread(self._make_request, method, endpoint, params)

    def is_connected(self) -> bool:
        if not self._connected:
            try:
                self._test_connection()
            except Exception as e:
                logger.warning(f"Connection verification failed: {e}")
                self._connected = False
        return self._connected and bool(self.api_key and self.api_secret)
    
    def _test_connection(self) -> bool:
        try:
            if not self.api_key or not self.api_secret:
                logger.error("API credentials missing in .env")
                self._connected = False
                return False
            result = self._make_request("GET", "/v5/market/time", {})
            self._connected = True
            logger.info(f"API connection test successful", extra={'endpoint': '/v5/market/time', 'environment': 'mainnet', 'account_type': self.account_type})
            return True
        except APIAuthenticationException as e:
            self._connected = False
            logger.error(f"API authentication failed during connection test: {str(e)}", extra={'error_type': 'authentication', 'environment': 'mainnet', 'account_type': self.account_type})
            return False
        except APIRateLimitException as e:
            self._connected = False
            logger.warning(f"Rate limit hit during connection test: {str(e)}", extra={'error_type': 'rate_limit', 'retry_after': e.retry_after})
            return False
        except APIException as e:
            self._connected = False
            logger.error(f"API error during connection test: {str(e)}", extra={'error_type': 'api_error', 'error_code': e.error_code})
            return False
        except Exception as e:
            self._connected = False
            logger.error(f"Unexpected error during connection test: {str(e)}", extra={'error_type': 'unexpected'})
            return False
    
    def get_connection_health(self) -> Dict[str, Any]:
        try:
            server_time = self._make_request("GET", "/v5/market/time", {}).get("time", 0)
            current_time = int(time.time() * 1000)
            time_diff = abs(current_time - int(server_time)) if server_time else float('inf')
            status = 'healthy' if time_diff < 5000 else 'degraded'
            self.last_successful_request = datetime.now()
        except Exception as e:
            logger.error(f"Bybit health check failed: {e}")
            status = 'unhealthy'
            time_diff = float('inf')
            server_time = 0
        health_info = {
            'connected': self._connected,
            'environment': 'mainnet',
            'account_type': self.account_type,
            'api_configured': bool(self.api_key and self.api_secret),
            'last_successful_request': self.last_successful_request.isoformat() if self.last_successful_request else None,
            'consecutive_errors': self.consecutive_errors,
            'status': status,
            'server_time': server_time,
            'local_time': current_time,
            'time_diff': time_diff,
            'statistics': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': round((self.successful_requests / max(self.total_requests, 1)) * 100, 2)
            },
            'rate_limiting': {
                'requests_per_second': self.rate_limit.requests_per_second,
                'requests_per_minute': self.rate_limit.requests_per_minute,
                'current_minute_count': self.rate_limit.minute_count
            }
        }
        return health_info

    def get_account_balance(self) -> Dict[str, 'WalletBalance']:
        from db import WalletBalance
        try:
            result = self._make_request("GET", "/v5/account/wallet-balance", {"accountType": self.account_type})
            balances: Dict[str, WalletBalance] = {}
            if result and "list" in result and result["list"]:
                wallet = result["list"][0]
                coins = wallet.get("coin", [])
                for coin in coins:
                    symbol = coin.get("coin", "")
                    if not symbol:
                        continue
                    available = float(coin.get("availableToWithdraw", 0) or 0.0)
                    total = float(coin.get("walletBalance", 0) or 0.0)
                    used = total - available
                    balances[symbol] = WalletBalance(
                        account_type=TRADING_MODE,
                        available=available,
                        used=used,
                        total=total,
                        currency=symbol,
                        exchange="bybit",
                        updated_at=datetime.utcnow(),
                    )
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance from Bybit: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        try:
            if symbol in self._price_cache:
                cache_time, price = self._price_cache[symbol]
                if time.time() - cache_time < 10:
                    return price
            # Public endpoint - no authentication required
            url = f"{self.base_url}/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get("retCode") == 0 and "result" in result and "list" in result["result"] and result["result"]["list"]:
                price = float(result["result"]["list"][0].get("lastPrice", 0))
                self._price_cache[symbol] = (time.time(), price)
                return price
            return 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_current_over_price(self, symbol: str) -> float:
        return self.get_current_price(symbol)

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        try:
            timeframe_map = {'1': '1', '5': '5', '15': '15', '30': '30', '60': '60', '240': '240', '1440': 'D'}
            timeframe = timeframe_map.get(interval, '60')
            # Public endpoint - no authentication required
            url = f"{self.base_url}/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": timeframe,
                "limit": str(limit)
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result.get("retCode") == 0 and "result" in result and "list" in result["result"]:
                klines = []
                for k in result["result"]["list"]:
                    klines.append({
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    })
                return sorted(klines, key=lambda x: x["timestamp"])
            return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        leverage: Optional[int] = 10,
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
            leverage = leverage or 10
            if TRADING_MODE == "virtual":
                entry_price = self.get_current_price(symbol)
                if entry_price <= 0:
                    raise ValueError(f"Invalid entry price for {symbol}: {entry_price}")
                order = {
                    "order_id": f"virtual_{int(time.time() * 1000)}",
                    "symbol": symbol,
                    "accountType": self.account_type,
                    "side": side,
                    "qty": qty,
                    "price": entry_price,
                    "status": "filled",
                    "timestamp": datetime.now(),
                    "virtual": True,
                    "leverage": leverage,
                    "stopLoss": str(stopLoss) if stopLoss is not None else None,
                    "takeProfit": str(takeProfit) if takeProfit is not None else None,
                    "margin_mode": mode.upper()
                }
                trade = Trade(
                    order_id=order["order_id"],
                    symbol=symbol,
                    exchange="bybit",
                    virtual=True,
                    side=side,
                    entry_price=entry_price,
                    qty=qty,
                    leverage=leverage,
                    sl=stopLoss,
                    tp=takeProfit,
                    indicators=indicators or {},
                    status="open",
                    created_at=datetime.now(timezone.utc)
                )
                db_manager.add_trade(trade)
                db_manager.session.commit()
                logger.info(f"Virtual order placed: {order['order_id']} - {symbol} {side} {qty}")
                return order
            if self.account_type == "UNIFIED":
                mode = "CROSS"
                logger.info(f"Unified account detected, using CROSS margin mode for {symbol}")
            else:
                trade_mode = 1 if mode.upper() == "ISOLATED" else 0
                lev_params = {
                    "category": "linear",
                    "symbol": symbol,
                    "tradeMode": trade_mode,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage)
                }
                await asyncio.to_thread(self._make_request, "POST", "/v5/position/switch-isolated", lev_params)
            entry_price = self.get_current_price(symbol)
            if entry_price <= 0:
                raise ValueError(f"Invalid entry price for {symbol}: {entry_price}")
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.title(),
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "IOC",
            }
            if stopLoss is not None:
                params["stopLoss"] = str(round(float(stopLoss), 6))
            if takeProfit is not None:
                params["takeProfit"] = str(round(float(takeProfit), 6))
            result = await asyncio.to_thread(self._make_request, "POST", "/v5/order/create", params)
            if result:
                order = {
                    "order_id": result.get("orderId"),
                    "symbol": symbol,
                    "accountType": self.account_type,
                    "side": side,
                    "qty": qty,
                    "price": entry_price,
                    "status": "pending",
                    "timestamp": datetime.now(),
                    "virtual": False,
                    "leverage": leverage,
                    "stopLoss": str(stopLoss) if stopLoss is not None else None,
                    "takeProfit": str(takeProfit) if takeProfit is not None else None,
                    "margin_mode": mode.upper()
                }
                trade = Trade(
                    order_id=order["order_id"],
                    symbol=symbol,
                    exchange="bybit",
                    virtual=False,
                    side=side,
                    entry_price=entry_price,
                    qty=qty,
                    leverage=leverage,
                    sl=stopLoss,
                    tp=takeProfit,
                    indicators=indicators or {},
                    status="open",
                    created_at=datetime.now(timezone.utc)
                )
                db_manager.add_trade(trade)
                db_manager.session.commit()
                logger.info(f"Order placed: {order['order_id']} - {symbol} {side} {qty}")
                return order
            return {}
        except APIException as e:
            if e.error_code == "100028":
                logger.warning(f"Unified account error for {symbol}: {e}. Using cross margin mode.")
                return await self.place_order(symbol, side, qty, leverage, mode="CROSS", stopLoss=stopLoss, takeProfit=takeProfit, indicators=indicators)
            logger.error(f"Error placing order for {symbol}: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error placing order for {symbol}: {e}")
            return {"error": str(e)}
        finally:
            db_manager.session.close()

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        db_manager = DatabaseManager()
        try:
            if TRADING_MODE == "virtual" and order_id.startswith("virtual_"):
                trade = db_manager.get_trade(order_id)
                if trade:
                    db_manager.update_trade(cast(int, trade.id), {"status": "canceled"})
                    logger.info(f"Virtual order canceled: {order_id} - {symbol}")
                    return True
                return False
            result = await asyncio.to_thread(self._make_request, "POST", "/v5/order/cancel", {
                "category": "linear",
                "symbol": symbol,
                "orderId": order_id
            })
            if result:
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
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
            result = self._make_request("GET", "/v5/order/realtime", params)
            if result and "list" in result:
                orders: List[Dict[str, Any]] = []
                for order in result["list"]:
                    orders.append({
                        "order_id": order.get("orderId"),
                        "symbol": order.get("symbol"),
                        "side": order.get("side").lower(),
                        "qty": float(order.get("qty", 0)),
                        "price": float(order.get("price", 0)),
                        "status": order.get("orderStatus").lower(),
                        "timestamp": datetime.fromtimestamp(int(order.get("createdTime", 0)) / 1000),
                        "virtual": False
                    })
                return orders
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        db_manager = DatabaseManager()
        try:
            params = {"category": "linear"}
            if symbol:
                params["symbol"] = symbol
            result = await asyncio.to_thread(self._make_request, "GET", "/v5/position/list", params)
            if result and "list" in result:
                positions: List[Dict] = []
                for pos in result["list"]:
                    size = float(pos.get("size", 0))
                    if size > 0:
                        position = {
                            "symbol": pos.get("symbol"),
                            "side": pos.get("side").lower(),
                            "size": size,
                            "entry_price": float(pos.get("avgPrice", 0)),
                            "mark_price": float(pos.get("markPrice", 0)),
                            "unrealized_pnl": float(pos.get("unrealisedPnl", 0)),
                            "leverage": float(pos.get("leverage", 10)),
                            "virtual": False
                        }
                        trade = db_manager.get_trade_by_position(pos.get("symbol"), "bybit", False)
                        if trade:
                            db_manager.update_trade(cast(int, trade.id), {
                                "pnl": position["unrealized_pnl"],
                                "status": "open"
                            })
                        positions.append(position)
                return positions
            return []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
        finally:
            db_manager.session.close()

    async def start_websocket(self, symbols: List[str]):
        try:
            if not self.loop:
                logger.error("Event loop not available")
                return
            async def websocket_handler():
                uri = f"{self.ws_url}/v5/public/linear"
                try:
                    async with websockets.connect(uri) as websocket:
                        self.ws_connection = websocket
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}" for symbol in symbols]
                        }
                        await websocket.send(json.dumps(subscribe_msg))
                        async for message in websocket:
                            data = json.loads(message)
                            if data.get("topic", "").startswith("tickers."):
                                ticker_data = data.get("data", {})
                                symbol = ticker_data.get("symbol")
                                price = float(ticker_data.get("lastPrice", 0))
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

    def get_trading_fees(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        try:
            result = self._make_request("GET", "/v5/account/fee-rate", {"category": "linear"})
            fees: Dict[str, Any] = {}
            if result and "list" in result:
                for item in result["list"]:
                    if symbol and item.get("symbol") != symbol:
                        continue
                    fees[item.get("symbol", "default")] = {
                        "maker_fee": float(item.get("makerFeeRate", 0)),
                        "taker_fee": float(item.get("takerFeeRate", 0))
                    }
            return fees if symbol else fees
        except Exception as e:
            logger.error(f"Error fetching Bybit trading fees: {e}")
            return {}

    def close(self):
        try:
            if self.ws_connection and self.loop:
                asyncio.run_coroutine_threadsafe(self.ws_connection.close(), self.loop)
            if self.session:
                self.session.close()
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            logger.info("Bybit client closed")
        except Exception as e:
            logger.error(f"Error closing client: {e}")