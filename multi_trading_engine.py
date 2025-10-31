# multi_trading_engine.py
import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import streamlit as st
from db import DatabaseManager, Trade, WalletBalance, User
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import (
    TradingException, APIException, APIRateLimitException,
    APIAuthenticationException, APITimeoutException, create_error_context
)
from utils import round_to_precision

logger = get_trading_logger('engine')


# ———————————————————————————————————————
# Unified Trading Engine
# ———————————————————————————————————————
class TradingEngine:
    def __init__(self, user_id: int, exchange: str = "bybit", account_type: str = "virtual"):
        self.user_id = user_id
        self.exchange = exchange.lower()
        self.account_type = account_type.lower()
        self.logger = get_trading_logger('engine')
        self.api_keys: Dict[str, Any] = {}

        self.binance_api_key: Optional[str] = None
        self.binance_api_secret: Optional[str] = None
        self.bybit_api_key: Optional[str] = None
        self.bybit_api_secret: Optional[str] = None

        # INIT DB FIRST
        self.db = DatabaseManager()
        self._load_api_keys()

        self.switch_exchange(self.exchange)

        if self.account_type not in ["virtual", "real"]:
            raise TradingException(f"Invalid account_type: {account_type}")

        self.settings = self._load_settings()
        self.exchange_name = self.exchange
        self._candle_cache: Dict[str, Tuple[datetime, List[Dict]]] = {}

        self._trading_enabled = True
        self._emergency_stop = False
        self._consecutive_failures = 0
        self._daily_pnl = 0.0
        self._daily_reset_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        self.max_position_size = float(self.settings.get("MAX_POSITION_SIZE", 10000.0))
        self.max_open_positions = int(self.settings.get("MAX_OPEN_POSITIONS", 10))
        self.max_daily_loss = float(self.settings.get("MAX_DAILY_LOSS", 1000.0))
        self.max_risk_per_trade = float(self.settings.get("MAX_RISK_PER_TRADE", 0.05))

        self.trade_count = 0
        self.successful_trades = 0
        self.failed_trades = 0

        self._initialize_wallets()

        logger.info("TradingEngine initialized", extra={
            "user_id": self.user_id,
            "exchange": self.exchange,
            "account_type": self.account_type,
        })

    # ----------------------------------------------------------------------
    # 1. Load API keys from the `users` table
    # ----------------------------------------------------------------------
    def _load_api_keys(self):
        try:
            with self.db.get_session() as s:
                user = s.query(User).filter_by(id=self.user_id).first()
                if not user:
                    return
                self.binance_api_key = user.binance_api_key
                self.binance_api_secret = user.binance_api_secret
                self.bybit_api_key = user.bybit_api_key
                self.bybit_api_secret = user.bybit_api_secret
        except Exception as e:
            self.logger.error(f"Failed to load API keys: {e}")

    # ----------------------------------------------------------------------
    # 2. Settings
    # ----------------------------------------------------------------------
    def _load_settings(self) -> Dict[str, Any]:
        base = load_settings(user_id=self.user_id) or {}
        exchange_settings = base.get(self.exchange, {})
        return {**base, **exchange_settings}

    # ----------------------------------------------------------------------
    # 3. Client initialisation (called from switch_exchange)
    # ----------------------------------------------------------------------
    def _init_client(self):
        if self.exchange == "bybit":
            from bybit_client import BybitClient
            return BybitClient
        elif self.exchange == "binance":
            from binance_client import BinanceClient
            return BinanceClient
        else:
            raise TradingException(f"Unsupported exchange: {self.exchange}")

    # ----------------------------------------------------------------------
    # 4. Update API credentials (Settings page → instant reload)
    # ----------------------------------------------------------------------
    def update_api_credentials(self, exchange: str, api_key: str, api_secret: str):
        exchange = exchange.lower()
        setattr(self, f"{exchange}_api_key", api_key)
        setattr(self, f"{exchange}_api_secret", api_secret)

        try:
            with self.db.get_session() as session:
                user = session.query(User).filter_by(id=self.user_id).first()
                if user:
                    setattr(user, f"{exchange}_api_key", api_key)
                    setattr(user, f"{exchange}_api_secret", api_secret)
                    session.commit()
        except Exception as e:
            self.logger.error(f"Failed to save API keys to DB: {e}")

        if self.exchange == exchange:
            self.switch_exchange(exchange)

    # ----------------------------------------------------------------------
    # 5. Switch exchange
    # ----------------------------------------------------------------------
    def switch_exchange(self, exchange: str):
        exchange = exchange.lower()
        if exchange not in ["binance", "bybit"]:
            raise ValueError("Unsupported exchange")

        self.exchange = exchange
        self.exchange_name = exchange
        self.logger.info(f"Switching exchange to: {exchange}")

        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass

        ClientClass = self._init_client()
        api_key = getattr(self, f"{exchange}_api_key")
        api_secret = getattr(self, f"{exchange}_api_secret")

        self.client = ClientClass(api_key, api_secret) if api_key and api_secret else None

        if st.session_state.get("current_exchange") != exchange:
            st.session_state.current_exchange = exchange

        self.reload_settings()

    # ----------------------------------------------------------------------
    # 6. Reload settings (after exchange or key change)
    # ----------------------------------------------------------------------
    def reload_settings(self):
        self.settings = self._load_settings()
        self.exchange_name = self.exchange

    # ----------------------------------------------------------------------
    # 7. Account type switch
    # ----------------------------------------------------------------------
    def set_account_type(self, account_type: str):
        account_type = account_type.lower()
        if account_type not in ["virtual", "real"]:
            raise ValueError("account_type must be 'virtual' or 'real'")
        self.account_type = account_type
        self.logger.info(f"Account type switched to: {account_type}")

    # ----------------------------------------------------------------------
    # Wallet Management
    # ----------------------------------------------------------------------
    def _initialize_wallets(self):
        for mode in ["virtual", "real"]:
            bal = self.db.get_wallet_balance(mode, self.user_id, self.exchange)
            if not bal:
                init = 10000.0 if mode == "virtual" else 0.0
                self.db.update_wallet_balance(
                    mode, self.user_id, init, 0.0, self.exchange
                )
                logger.info(f"Initialized {mode} wallet: ${init:.2f}")

    def get_balance(self) -> Dict[str, float]:
        w = self.db.get_wallet_balance(self.account_type, self.user_id, self.exchange)
        if not w:
            return {"total": 0.0, "available": 0.0, "used": 0.0}
        return {
            "total": float(w.total or 0.0),
            "available": float(w.available or 0.0),
            "used": float(w.used or 0.0),
        }

    def _update_wallet(
        self,
        available_delta: float = 0.0,
        used_delta: float = 0.0,
        capital_delta: float = 0.0,
    ):
        self.db.update_wallet_balance(
            self.account_type,
            self.user_id,
            available_delta=available_delta,
            used_delta=used_delta,
            capital_delta=capital_delta,
            exchange=self.exchange,
        )

    # ----------------------------------------------------------------------
    # Trading State
    # ----------------------------------------------------------------------
    def _reset_daily_stats(self):
        now = datetime.now(timezone.utc)
        if now >= self._daily_reset_time + timedelta(days=1):
            self._daily_pnl = 0.0
            self._daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            logger.info("Daily PnL reset")

    def _check_emergency_conditions(self) -> bool:
        self._reset_daily_stats()
        if self._daily_pnl <= -self.max_daily_loss:
            self.emergency_stop("Daily loss limit exceeded")
            return False
        if self._consecutive_failures >= 10:
            self.emergency_stop("Too many consecutive failures")
            return False
        if self.client and not self.client.is_connected():
            logger.warning("Client disconnected")
            return False
        return True

    def is_trading_enabled(self) -> bool:
        return (
            self._trading_enabled
            and not self._emergency_stop
            and self._check_emergency_conditions()
        )

    def enable_trading(self) -> bool:
        if self._emergency_stop:
            logger.warning("Cannot enable: emergency stop active")
            return False
        self._trading_enabled = True
        logger.info("Trading enabled")
        return True

    def disable_trading(self, reason: str = "Manual") -> bool:
        self._trading_enabled = False
        logger.warning(f"Trading disabled: {reason}")
        return True

    def emergency_stop(self, reason: str = "Emergency") -> bool:
        self._emergency_stop = True
        self._trading_enabled = False
        logger.critical(f"EMERGENCY STOP: {reason}")
        for trade in self.get_open_trades(virtual=False):
            try:
                asyncio.create_task(self._close_position(trade))
            except Exception as e:
                logger.error(f"Failed to close {trade.symbol}: {e}")
        return True

    async def _close_position(self, trade: Trade):
        close_side = "Sell" if trade.side.upper() in ["BUY", "LONG"] else "Buy"
        await self.client.place_order(
            symbol=trade.symbol,
            side=close_side,
            qty=trade.qty,
            leverage=trade.leverage,
            mode="CROSS",
        )
        self.db.update_trade(trade.id, {"status": "closed", "closed_at": datetime.now(timezone.utc)})
        logger.info(f"Closed {trade.symbol} during emergency stop")

    # ----------------------------------------------------------------------
    # Market Data
    # ----------------------------------------------------------------------
    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        cache_key = f"{symbol}_{interval}_{limit}"
        now = datetime.now(timezone.utc)
        if cache_key in self._candle_cache:
            cached_time, data = self._candle_cache[cache_key]
            if (now - cached_time).total_seconds() < 300:
                return data
        try:
            klines = self.client.get_klines(symbol, interval, limit) if self.client else []
            if klines:
                self._candle_cache[cache_key] = (now, klines)
            return klines or []
        except Exception as e:
            logger.error(f"Failed to fetch candles {symbol}: {e}")
            return []

    # ----------------------------------------------------------------------
    # Position Sizing
    # ----------------------------------------------------------------------
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        risk_pct: Optional[float] = None,
        leverage: Optional[int] = None,
    ) -> float:
        try:
            risk_pct = risk_pct or self.max_risk_per_trade
            lev = leverage or self.settings.get("LEVERAGE", 15)
            bal = self.get_balance()["available"]
            if bal <= 0:
                return 0.0
            risk_amount = max(bal * risk_pct, 2.0)
            pos_value = risk_amount * lev
            size = pos_value / entry_price

            info = self.client.get_symbol_info(symbol) if self.client else {}
            lot = info.get("lotSizeFilter", {})
            min_qty = float(lot.get("minOrderQty", 0))
            qty_step = float(lot.get("qtyStep", 0))

            if min_qty > 0 and size < min_qty:
                margin = min_qty * entry_price / lev
                if margin > bal:
                    return 0.0
                size = min_qty
            if qty_step > 0:
                size = round(size / qty_step) * qty_step

            return round_to_precision(size, 8)
        except Exception as e:
            logger.error(f"Position size error {symbol}: {e}")
            return 0.0

    # ----------------------------------------------------------------------
    # Trade Execution
    # ----------------------------------------------------------------------
    async def execute_trade(self, signal: Dict[str, Any]) -> bool:
        if not self.is_trading_enabled():
            logger.warning("Trading disabled")
            return False

        symbol = signal.get("symbol", "").replace("/", "")
        side = signal.get("side", "Buy").upper()
        entry_price = float(signal.get("entry") or (self.client.get_current_price(symbol) if self.client else 0))
        if entry_price <= 0:
            return False

        qty = self.calculate_position_size(symbol, entry_price)
        if qty <= 0:
            return False

        if len(self.get_open_trades(virtual=False)) >= self.max_open_positions:
            logger.warning("Max open positions reached")
            return False

        trade = self._create_trade(signal, qty, entry_price)
        if not self.db.add_trade(trade):
            return False

        if self.account_type == "virtual":
            self.db.update_trade(trade.id, {"status": "open"})
            self._update_wallet(used_delta=qty * entry_price)
            logger.info(f"Virtual {side} {symbol} @ {entry_price:.2f}")
            self.successful_trades += 1
            return True

        return await self._execute_real_order(trade, signal)

    async def _execute_real_order(self, trade: Trade, signal: Dict) -> bool:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                order = await self.client.place_order(
                    symbol=trade.symbol,
                    side=trade.side,
                    qty=trade.qty,
                    leverage=signal.get("leverage", 15),
                    mode="CROSS",
                    stop_loss=signal.get("sl"),
                    take_profit=signal.get("tp"),
                )
                if self._is_order_success(order):
                    order_id = self._extract_order_id(order)
                    self.db.update_trade(trade.id, {"order_id": order_id, "status": "open"})
                    self._update_wallet(used_delta=trade.qty * trade.entry_price)
                    logger.info(f"Real trade placed: {order_id}")
                    self.successful_trades += 1
                    self._consecutive_failures = 0
                    return True
                else:
                    err = order.get("error") or order.get("retMsg", "Unknown")
                    self._handle_failure(trade.id, err)
                    return False
            except (APIRateLimitException, APITimeoutException):
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                self._handle_failure(trade.id, "Rate limit/timeout")
                return False
            except Exception as e:
                self._handle_failure(trade.id, str(e))
                return False
        return False

    def _create_trade(self, signal: Dict, qty: float, entry_price: float) -> Trade:
        return Trade(
            user_id=self.user_id,
            exchange=self.exchange,
            symbol=signal["symbol"],
            side=signal.get("side", "Buy").upper(),
            qty=qty,
            entry_price=entry_price,
            status="pending",
            virtual=(self.account_type == "virtual"),
            leverage=signal.get("leverage", 15),
            sl=signal.get("sl"),
            tp=signal.get("tp"),
            trail=signal.get("trail"),
            margin_usdt=signal.get("margin_usdt"),
            signal_id=signal.get("id"),
            indicators=signal.get("indicators", {}),
        )

    def _is_order_success(self, order: Dict) -> bool:
        if self.exchange == "bybit":
            return order.get("retCode") == 0
        if self.exchange == "binance":
            return "error" not in order and order.get("status") in ["NEW", "FILLED"]
        return False

    def _extract_order_id(self, order: Dict) -> str:
        if self.exchange == "bybit":
            return order.get("result", {}).get("orderId", "")
        if self.exchange == "binance":
            return str(order.get("order_id", ""))
        return ""

    def _handle_failure(self, trade_id: int, error: str):
        self.db.update_trade(trade_id, {"status": "failed", "error_message": error})
        self.failed_trades += 1
        self._consecutive_failures += 1

    # ----------------------------------------------------------------------
    # Trade Management
    # ----------------------------------------------------------------------
    def get_open_trades(self, virtual: Optional[bool] = None) -> List[Trade]:
        vf = (self.account_type == "virtual") if virtual is None else virtual
        return self.db.get_open_trades(virtual=vf, exchange=self.exchange, user_id=self.user_id)

    def get_closed_trades(self, limit: int = 100) -> List[Trade]:
        return self.db.get_trades(
            virtual=(self.account_type == "virtual"),
            exchange=self.exchange,
            user_id=self.user_id,
            limit=limit,
        )

    def sync_real_positions(self) -> bool:
        if self.account_type != "real" or self.exchange != "bybit":
            return False
        try:
            positions = self.client._make_request(
                "GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"}
            )["list"]
            for pos in positions:
                if float(pos.get("size", 0)) <= 0:
                    continue
                # Add sync logic here if needed
                pass
            return True
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False

    # ----------------------------------------------------------------------
    # Virtual PnL
    # ----------------------------------------------------------------------
    def calculate_virtual_pnl(self, trade: Trade) -> float:
        try:
            cur = self.client.get_current_price(trade.symbol) if self.client else 0
            if trade.side in ["BUY", "LONG"]:
                return (cur - trade.entry_price) * trade.qty
            else:
                return (trade.entry_price - cur) * trade.qty
        except Exception:
            return 0.0

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    def close(self):
        if hasattr(self.client, "close"):
            self.client.close()
        logger.info("TradingEngine closed")