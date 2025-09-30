import os
import hmac
import hashlib
import time
import json
import asyncio
import websockets
from typing import Dict, Any, Optional, List
import requests.adapters
from datetime import datetime
import requests
import threading
from dataclasses import dataclass
from logging_config import get_trading_logger
from exceptions import (
    APIException, APIConnectionException, APIRateLimitException, 
    APIAuthenticationException, APITimeoutException, APIDataException,
    APIErrorRecoveryStrategy, create_error_context
)

logger = get_trading_logger('api_binance_client')

@dataclass
class RateLimitInfo:
    requests_per_second: int = 10
    requests_per_minute: int = 1200
    last_request_time: float = 0
    request_count: int = 0
    minute_start: float = 0
    minute_count: int = 0

class BinanceClient:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.account_type = os.getenv("BINANCE_ACCOUNT_TYPE", "SPOT").upper()  # optional: SPOT, MARGIN, FUTURES

        # Mainnet only
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443"

        
        # Connection and session management
        self.session: Optional[requests.Session] = None
        self.ws_connection = None
        self.loop = None
        self._price_cache = {}
        self._connected = False
        self._connection_lock = threading.Lock()
        
        # Rate limiting
        self.rate_limit = RateLimitInfo()
        self.rate_limit_lock = threading.Lock()
        
        # Error handling
        self.recovery_strategy = APIErrorRecoveryStrategy(max_retries=3, delay=1.0)
        self.last_error_time = None
        self.consecutive_errors = 0
        
        # Timeout configuration
        self.request_timeout = 30  # seconds
        self.connect_timeout = 10  # seconds
        
        # Health monitoring
        self.last_successful_request = None
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Initialize connection
        self._initialize_session()
        self._test_connection()
        
        # Start background event loop for WebSocket
        self._start_background_loop()
        
        logger.info(f"BinanceClient initialized - Environment: mainnet - Account Type: {self.account_type}")
    
    def _initialize_session(self):
        """Initialize HTTP session with proper configuration"""
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
            error_context = create_error_context(
                module=__name__,
                function='_initialize_session'
            )
            raise APIConnectionException(
                f"Failed to initialize HTTP session: {str(e)}",
                endpoint='session_init',
                context=error_context,
                original_exception=e
            )

    def _start_background_loop(self):
        """Start background event loop for async operations"""
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

    def _generate_signature(self, params: str) -> str:
        """Generate API signature"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def _get_headers(self) -> Dict[str, str]:
        """Get authenticated headers"""
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }

    def _check_rate_limit(self) -> bool:
        """Check and enforce rate limits"""
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
        """Validate API credentials are present"""
        if not self.api_key or not self.api_secret:
            raise APIAuthenticationException(
                "API credentials not configured. Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables.",
                context=create_error_context(module=__name__, function='_validate_api_credentials')
            )
        return True
    
    def _handle_api_error(self, response_data: Dict, endpoint: str) -> None:
        """Handle API error responses"""
        code = response_data.get("code", -1)
        msg = response_data.get("msg", "Unknown error")
        
        if code == -1003:
            raise APIRateLimitException(
                f"Rate limit exceeded: {msg}",
                retry_after=60,
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        elif code == -1022:
            raise APIAuthenticationException(
                f"Invalid API key or signature: {msg}",
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        elif code == -2015:
            raise APIAuthenticationException(
                f"Invalid signature: {msg}",
                context=create_error_context(module=__name__, function='_handle_api_error')
            )
        else:
            raise APIException(
                f"API Error (code {code}): {msg}",
                error_code=str(code),
                context=create_error_context(
                    module=__name__, 
                    function='_handle_api_error',
                    extra_data={'endpoint': endpoint, 'code': code}
                )
            )
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated API request with comprehensive error handling"""
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
                    
                    timestamp = str(int(time.time() * 1000))
                    params["timestamp"] = timestamp
                    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                    signature = self._generate_signature(query_string)
                    params["signature"] = signature
                    
                    headers = self._get_headers()
                    
                    if method.upper() == "GET":
                        response = self.session.get(
                            url, 
                            params=params, 
                            headers=headers, 
                            timeout=(self.connect_timeout, self.request_timeout)
                        )
                    else:
                        response = self.session.post(
                            url, 
                            json=params, 
                            headers=headers, 
                            timeout=(self.connect_timeout, self.request_timeout)
                        )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise APIRateLimitException(
                        f"Rate limit exceeded on {endpoint}",
                        retry_after=retry_after,
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'attempt': attempt}
                        )
                    )
                
                response.raise_for_status()
                
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    raise APIDataException(
                        f"Invalid JSON response from {endpoint}: {str(e)}",
                        response_data=response.text[:1000],
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': response.status_code}
                        )
                    )
                
                if "code" in data and data["code"] != 0:
                    self._handle_api_error(data, endpoint)
                
                self.successful_requests += 1
                self.last_successful_request = datetime.now()
                self.consecutive_errors = 0
                
                logger.info(
                    f"API request successful: {method} {endpoint}",
                    extra={
                        'endpoint': endpoint,
                        'method': method,
                        'response_time_ms': round(response_time, 2),
                        'status_code': response.status_code,
                        'attempt': attempt + 1
                    }
                )
                
                return data
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                
                if not self.recovery_strategy.should_retry(e, attempt):
                    raise APITimeoutException(
                        f"Request timeout for {endpoint} after {attempt + 1} attempts: {str(e)}",
                        timeout_duration=self.request_timeout,
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'total_attempts': attempt + 1}
                        ),
                        original_exception=e
                    )
                
                retry_delay = self.recovery_strategy.get_delay(attempt)
                logger.warning(
                    f"API request failed (attempt {attempt + 1}), retrying in {retry_delay}s: {str(e)}",
                    extra={'endpoint': endpoint, 'attempt': attempt + 1, 'retry_delay': retry_delay}
                )
                time.sleep(retry_delay)
                
            except requests.exceptions.HTTPError as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                
                status_code = e.response.status_code if e.response else None
                
                if status_code == 401:
                    raise APIAuthenticationException(
                        f"Authentication failed for {endpoint}: {str(e)}",
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': status_code}
                        ),
                        original_exception=e
                    )
                elif status_code == 403:
                    raise APIAuthenticationException(
                        f"Access forbidden for {endpoint}: {str(e)}",
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': status_code}
                        ),
                        original_exception=e
                    )
                else:
                    raise APIConnectionException(
                        f"HTTP error for {endpoint}: {str(e)}",
                        endpoint=endpoint,
                        status_code=status_code,
                        context=create_error_context(
                            module=__name__,
                            function='_make_request',
                            extra_data={'endpoint': endpoint, 'status_code': status_code}
                        ),
                        original_exception=e
                    )
            
            except Exception as e:
                self.failed_requests += 1
                self.consecutive_errors += 1
                
                logger.error(
                    f"Unexpected error in API request to {endpoint}: {str(e)}",
                    extra={'endpoint': endpoint, 'attempt': attempt + 1}
                )
                
                raise APIException(
                    f"Unexpected error for {endpoint}: {str(e)}",
                    context=create_error_context(
                        module=__name__,
                        function='_make_request',
                        extra_data={'endpoint': endpoint, 'attempt': attempt + 1}
                    ),
                    original_exception=e
                )
        
        raise APIException(
            f"Maximum retry attempts exceeded for {endpoint}",
            context=create_error_context(
                module=__name__,
                function='_make_request',
                extra_data={'endpoint': endpoint, 'max_retries': self.recovery_strategy.max_retries}
            )
        )

    def is_connected(self) -> bool:
        """Check if client is connected and authenticated"""
        if not self._connected:
            try:
                self._test_connection()
            except Exception as e:
                logger.warning(f"Connection verification failed: {e}")
                self._connected = False
        return self._connected and bool(self.api_key and self.api_secret)
    
    def _test_connection(self) -> bool:
        """Test API connection with comprehensive error handling"""
        try:
            if not self.api_key or not self.api_secret:
                logger.error("API credentials missing in .env")
                self._connected = False
                return False

            result = self._make_request("GET", "/api/v3/ping", {})
            self._connected = True
            
            logger.info(
                "API connection test successful",
                extra={
                    'endpoint': '/api/v3/ping',
                    'environment': 'mainnet'
                }
            )

            return True
            
        except APIAuthenticationException as e:
            self._connected = False
            logger.error(
                f"API authentication failed during connection test: {str(e)}",
                extra={'error_type': 'authentication', 'environment': 'mainnet'}
            )
            return False
            
        except APIRateLimitException as e:
            self._connected = False
            logger.warning(
                f"Rate limit hit during connection test: {str(e)}",
                extra={'error_type': 'rate_limit', 'retry_after': e.retry_after}
            )
            return False
            
        except APIException as e:
            self._connected = False
            logger.error(
                f"API error during connection test: {str(e)}",
                extra={'error_type': 'api_error', 'error_code': e.error_code}
            )
            return False
            
        except Exception as e:
            self._connected = False
            logger.error(
                f"Unexpected error during connection test: {str(e)}",
                extra={'error_type': 'unexpected'}
            )
            return False
    
    def get_connection_health(self) -> Dict[str, Any]:
        """Get comprehensive connection health information"""
        health_info = {
            'connected': self._connected,
            'environment': 'mainnet',
            'api_configured': bool(self.api_key and self.api_secret),
            'last_successful_request': self.last_successful_request.isoformat() if self.last_successful_request else None,
            'consecutive_errors': self.consecutive_errors,
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
        
        if not health_info['api_configured']:
            health_info['status'] = 'unconfigured'
        elif not health_info['connected']:
            health_info['status'] = 'disconnected'
        elif self.consecutive_errors > 5:
            health_info['status'] = 'degraded'
        else:
            health_info['status'] = 'healthy'
            
        return health_info
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Get exchange information (symbols, filters, trading rules)"""
        try:
            endpoint = "/api/v3/exchangeInfo"
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()

            # Public endpoint, so no signature needed
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=(self.connect_timeout, self.request_timeout))
            response.raise_for_status()
            data = response.json()

            return data

        except Exception as e:
            logger.error(f"Error fetching exchange info: {e}")
            return {}


    from typing import TYPE_CHECKING, Dict
    if TYPE_CHECKING:
        from db import WalletBalance
    def get_account_balance(self) -> "Dict[str, 'WalletBalance']":
        from db import WalletBalance
        """Get account wallet balance as WalletBalance objects keyed by coin symbol"""
        try:
            result = self._make_request("GET", "/api/v3/account", {})

            balances: Dict[str, "WalletBalance"] = {}

            if result and "balances" in result:
                for asset in result["balances"]:
                    symbol = asset.get("asset", "")
                    if not symbol:
                        continue

                    available = float(asset.get("free", 0))
                    locked = float(asset.get("locked", 0))
                    total = available + locked

                    balances[symbol] = WalletBalance(
                        trading_mode="real",
                        capital=total,
                        available=available,
                        used=locked,
                        start_balance=total,
                        currency=symbol,
                        updated_at=datetime.utcnow(),
                    )

            return balances

        except Exception as e:
            logger.error(f"Error getting account balance from Binance: {e}")
            return {}

    def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        try:
            if symbol in self._price_cache:
                cache_time, price = self._price_cache[symbol]
                if time.time() - cache_time < 10:
                    return price

            result = self._make_request("GET", "/api/v3/ticker/price", {"symbol": symbol})
            
            if result and "price" in result:
                price = float(result["price"])
                self._price_cache[symbol] = (time.time(), price)
                return price
            return 0.0
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_current_over_price(self, symbol: str) -> float:
        """Alias for get_current_price for backward compatibility"""
        return self.get_current_price(symbol)

    def get_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Get historical kline/candlestick data"""
        try:
            result = self._make_request("GET", "/api/v3/klines", {
                "symbol": symbol,
                "interval": interval,
                "limit": str(limit)
            })
            
            if result:
                klines = []
                for k in result:
                    klines.append({
                        "time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5])
                    })
                return sorted(klines, key=lambda x: x["time"])
            return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    async def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                         price: Optional[float] = None, stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Dict:
        """Place a trading order"""
        try:
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": str(qty),
                "timeInForce": "GTC"
            }
            
            if price and order_type.lower() == "limit":
                params["price"] = str(price)
            
            if stop_loss:
                params["stopPrice"] = str(stop_loss)
                params["type"] = "STOP_LOSS_LIMIT"
            
            if take_profit:
                params["takeProfit"] = str(take_profit)
                params["type"] = "TAKE_PROFIT_LIMIT"

            result = self._make_request("POST", "/api/v3/order", params)
            
            if result:
                return {
                    "order_id": str(result.get("orderId")),
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price or self.get_current_price(symbol),
                    "status": result.get("status", "pending"),
                    "timestamp": datetime.now(),
                    "virtual": False
                }
            return {}
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        try:
            result = self._make_request("DELETE", "/api/v3/order", {
                "symbol": symbol,
                "orderId": order_id
            })
            return bool(result)
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
                
            result = self._make_request("GET", "/api/v3/openOrders", params)
            
            if result:
                orders = []
                for order in result:
                    orders.append({
                        "order_id": str(order.get("orderId")),
                        "symbol": order.get("symbol"),
                        "side": order.get("side"),
                        "qty": float(order.get("origQty", 0)),
                        "price": float(order.get("price", 0)),
                        "status": order.get("status"),
                        "timestamp": datetime.fromtimestamp(int(order.get("time", 0)) / 1000)
                    })
                return orders
            return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions"""
        try:
            balances = self.get_account_balance()
            positions = []
            
            for symbol, balance in balances.items():
                if balance.used > 0:
                    price = self.get_current_price(symbol + "USDT")
                    positions.append({
                        "symbol": symbol,
                        "side": "LONG" if balance.used > 0 else "SHORT",
                        "size": balance.used,
                        "entry_price": price,
                        "mark_price": price,
                        "unrealized_pnl": 0.0,
                        "leverage": 1.0
                    })
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def start_websocket(self, symbols: List[str]):
        """Start WebSocket connection for real-time data"""
        try:
            if not self.loop:
                logger.error("Event loop not available")
                return

            async def websocket_handler():
                streams = "/".join([f"{symbol.lower()}@ticker" for symbol in symbols])
                uri = f"{self.ws_url}/ws/{streams}"
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
        """Close connections and cleanup"""
        try:
            if self.ws_connection and self.loop:
                asyncio.run_coroutine_threadsafe(self.ws_connection.close(), self.loop)
            if self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            logger.info("Binance client closed")
        except Exception as e:
            logger.error(f"Error closing client: {e}")