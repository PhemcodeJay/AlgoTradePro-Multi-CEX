import os
import json
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from db import db_manager, Trade, WalletBalance
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import TradingException, create_error_context, APIException, APIRateLimitException, APIAuthenticationException, APITimeoutException

logger = get_trading_logger('engine')

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()


def to_float(value: Any) -> float:
    """
    Safely convert value to float.
    Returns 0.0 for None, non-convertible values, or SQLAlchemy Column objects.
    """
    try:
        if value is None:
            return 0.0
        # If it's already numeric
        if isinstance(value, (int, float)):
            return float(value)
        # Try cast (handles strings containing numbers)
        return float(value)
    except Exception:
        # Defensive fallback for SQLAlchemy ColumnElements or unsupported types
        return 0.0


def is_true(value: Any) -> bool:
    """
    Safely evaluate truthiness without calling __bool__ on SQLAlchemy ColumnElement.
    Returns False for None or non-boolean-like values.
    """
    try:
        if isinstance(value, bool):
            return value
        # numeric truthiness
        if isinstance(value, (int, float)):
            return value != 0
        if value is None:
            return False
        # Try to interpret strings like "True", "False"
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "y")
        # Fallback: attempt bool conversion inside try (may raise for ColumnElement)
        return bool(value)
    except Exception:
        return False


class TradingEngine:
    def __init__(self):
        try:
            # Load base + exchange-specific settings
            base_settings = load_settings()
            self.exchange_name = base_settings.get("EXCHANGE", "binance").lower()
            exchange_overrides = base_settings.get(self.exchange_name, {})

            # Merge exchange overrides into settings
            self.settings = {**base_settings, **exchange_overrides}

            # Dynamically import correct client
            if self.exchange_name == "binance":
                from binance_client import BinanceClient
                self.client = BinanceClient()
            elif self.exchange_name == "bybit":
                from bybit_client import BybitClient
                self.client = BybitClient()
            else:
                raise TradingException(f"Unsupported exchange: {self.exchange_name}")

            self.db = db_manager
            self._candle_cache: Dict[str, Any] = {}

            # Position safety limits
            self.max_position_size = to_float(self.settings.get("MAX_POSITION_SIZE", 10000.0))
            self.max_open_positions = int(self.settings.get("MAX_OPEN_POSITIONS", 10))
            self.max_daily_loss = to_float(self.settings.get("MAX_DAILY_LOSS", 1000.0))
            self.max_risk_per_trade = to_float(self.settings.get("MAX_RISK_PER_TRADE", 0.05))

            # Trading state management
            self._trading_enabled = True
            self._emergency_stop = False
            self._last_health_check: Optional[datetime] = None
            self._consecutive_failures = 0
            self._daily_pnl = 0.0
            self._daily_reset_time = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Performance tracking
            self.trade_count = 0
            self.successful_trades = 0
            self.failed_trades = 0

            logger.info(
                f"Trading engine initialized for {self.exchange_name}",
                extra={
                    'exchange': self.exchange_name,
                    'max_position_size': self.max_position_size,
                    'max_open_positions': self.max_open_positions,
                    'max_daily_loss': self.max_daily_loss
                }
            )

        except Exception as e:
            error_context = create_error_context(
                module=__name__,
                function='__init__',
                extra_data={'settings': base_settings if 'base_settings' in locals() else None}
            )
            logger.error(f"Failed to initialize trading engine: {str(e)}", extra=error_context)
            raise TradingException(f"Failed to initialize trading engine: {str(e)}", context=error_context)

    def switch_exchange(self, exchange: str) -> bool:
        """Switch the trading engine to a different exchange"""
        try:
            exchange = exchange.lower()
            if exchange not in ["binance", "bybit"]:
                logger.error(f"Unsupported exchange: {exchange}")
                return False

            if exchange == self.exchange_name:
                logger.info(f"Already using {exchange}, no switch needed")
                return True

            # Update settings
            self.exchange_name = exchange
            base_settings = load_settings()
            exchange_overrides = base_settings.get(exchange, {})
            self.settings = {**base_settings, **exchange_overrides}

            # Reinitialize client
            if exchange == "binance":
                from binance_client import BinanceClient
                self.client = BinanceClient()
            elif exchange == "bybit":
                from bybit_client import BybitClient
                self.client = BybitClient()

            logger.info(f"Switched to {exchange} exchange")
            return True

        except Exception as e:
            logger.error(f"Error switching to {exchange}: {str(e)}")
            return False

    def enable_trading(self) -> bool:
        """Enable trading"""
        try:
            if is_true(self._emergency_stop):
                logger.warning("Cannot enable trading: emergency stop is active")
                return False
            self._trading_enabled = True
            logger.info("Trading enabled")
            return True
        except Exception as e:
            logger.error(f"Error enabling trading: {str(e)}")
            return False

    def disable_trading(self, reason: str = "Manual stop") -> bool:
        """Disable trading"""
        try:
            self._trading_enabled = False
            logger.info(f"Trading disabled: {reason}")
            return True
        except Exception as e:
            logger.error(f"Error disabling trading: {str(e)}")
            return False

    def emergency_stop(self, reason: str = "Emergency stop") -> bool:
        """Trigger an emergency stop"""
        try:
            self._emergency_stop = True
            self._trading_enabled = False
            logger.error(f"Emergency stop triggered: {reason}")
            return True
        except Exception as e:
            logger.error(f"Error triggering emergency stop: {str(e)}")
            return False

    def get_trade_statistics(self, account_type: str) -> Dict[str, Any]:
        """Get trade statistics for the given account type"""
        try:
            is_virtual = account_type.lower() == "virtual"
            trades = self.db.get_trades(virtual=is_virtual, exchange=self.exchange_name) or []
            total_trades = len(trades)
            successful_trades = len([
                t for t in trades
                if getattr(t, "status", None) == "closed" and to_float(getattr(t, "pnl", 0)) > 0.0
            ])
            total_pnl = sum(to_float(getattr(t, "pnl", 0)) for t in trades if getattr(t, "status", None) == "closed")
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0.0

            return {
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl
            }
        except Exception as e:
            logger.error(f"Error fetching trade statistics: {str(e)}")
            return {
                "total_trades": 0,
                "successful_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0
            }

    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled"""
        try:
            if is_true(self._emergency_stop):
                return False
            # some clients may provide method is_connected; check carefully
            try:
                client_connected = getattr(self.client, "is_connected", None)
                if callable(client_connected):
                    if not client_connected():
                        logger.warning("Client not connected, trading disabled")
                        return False
                else:
                    # If there's an attribute `.connected` (bool-like), check it safely
                    connected_attr = getattr(self.client, "connected", None)
                    if not is_true(connected_attr):
                        logger.warning("Client not connected (attribute check), trading disabled")
                        return False
            except Exception:
                # If any error evaluating connectivity, default to disabled (safer)
                logger.warning("Unable to determine client connectivity; treating as not connected")
                return False

            if to_float(self._daily_pnl) <= -to_float(self.max_daily_loss):
                logger.warning(f"Daily loss limit reached: {self._daily_pnl}")
                return False
            return bool(self._trading_enabled)
        except Exception as e:
            logger.error(f"Error checking trading status: {e}")
            return False

    def get_settings(self) -> Tuple[int, int]:
        """Return trading settings"""
        return (
            int(self.settings.get("SCAN_INTERVAL", 3600)),
            int(self.settings.get("TOP_N_SIGNALS", 10))
        )

    def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            symbol = symbol.replace('/', '')  # Ensure symbol format
            balance = self.get_account_balance() or {}
            usdt_balance = to_float(balance.get("USDT", {}).get("free", 0))
            if usdt_balance <= 0:
                logger.warning("No available USDT balance for position sizing")
                return 0.0

            risk_amount = usdt_balance * to_float(self.max_risk_per_trade)
            atr = self.get_atr(symbol, "60")  # Use 1h ATR
            if atr <= 0:
                logger.warning(f"Invalid ATR for {symbol}, using default size")
                return risk_amount / entry_price if entry_price > 0 else 0.0

            position_size = risk_amount / atr
            position_value = position_size * entry_price

            if position_value > to_float(self.max_position_size):
                position_size = to_float(self.max_position_size) / entry_price if entry_price > 0 else 0.0

            return round(position_size, 6)
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def get_atr(self, symbol: str, interval: str) -> float:
        """Get ATR for a symbol"""
        try:
            symbol = symbol.replace('/', '')  # Ensure symbol format
            klines = self.client.get_klines(symbol, interval, limit=14)
            if not klines:
                return 0.0
            highs = [to_float(k.get("high", 0)) for k in klines]
            lows = [to_float(k.get("low", 0)) for k in klines]
            closes = [to_float(k.get("close", 0)) for k in klines]
            trs = []
            for i in range(1, len(klines)):
                high_low = highs[i] - lows[i]
                high_prev_close = abs(highs[i] - closes[i - 1])
                low_prev_close = abs(lows[i] - closes[i - 1])
                tr = max(high_low, high_prev_close, low_prev_close)
                trs.append(tr)
            return float(np.mean(trs)) if trs else 0.0
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0

    async def check_position_limits(self, symbol: str) -> bool:
        """Check if new position can be opened"""
        try:
            symbol = symbol.replace('/', '')  # Ensure symbol format
            open_positions = await self.client.get_positions(symbol) or []
            if len(open_positions) >= int(self.max_open_positions):
                logger.warning(f"Max open positions reached: {len(open_positions)}")
                return False
            # defensively compute numeric values
            total_position_value = 0.0
            for p in open_positions:
                size = to_float(p.get("size", 0))
                entry_price = to_float(p.get("entry_price", 0))
                total_position_value += size * entry_price
            if total_position_value >= to_float(self.max_position_size):
                logger.warning(f"Max position size reached: {total_position_value}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking position limits for {symbol}: {e}")
            return False

    async def execute_trade(self, signal: Dict[str, Any]) -> bool:
        """Execute a trade based on signal with enhanced robustness"""
        try:
            symbol = signal.get("symbol", '').replace('/', '')
            side = signal.get("side", "BUY").lower()
            entry_price = to_float(signal.get("entry", 0))
            qty = self.calculate_position_size(symbol, entry_price)
            leverage = signal.get("leverage", 10)
            mode = signal.get("mode", "CROSS")
            stop_loss = signal.get("sl")
            take_profit = signal.get("tp")
            indicators = signal.get("indicators", {})

            if not await self.check_position_limits(symbol):
                logger.warning(f"Position limits exceeded for {symbol}")
                return False

            if entry_price <= 0 or qty <= 0:
                logger.error(f"Invalid trade parameters: price={entry_price}, qty={qty}")
                return False

            if TRADING_MODE == "virtual":
                # If virtual mode, update capital and execute virtual trade
                self._update_capital_json("virtual", qty * entry_price)
                return self.execute_virtual_trade(signal)

            # Create trade entry in database
            trade = Trade(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=entry_price,
                status="pending",
                virtual=False,
                leverage=leverage,
                sl=stop_loss,
                tp=take_profit,
                trail=signal.get("trail"),
                margin_usdt=signal.get("margin_usdt"),
                exchange=self.exchange_name,
                signal_id=signal.get("id"),
                indicators=indicators
            )
            self.db.add_trade(trade)
            self.db.session.commit()

            # Update capital for real trades
            self._update_capital_json("real", qty * entry_price)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    place_order = self.client.place_order
                    place_args = (symbol, side, qty, leverage, mode, stop_loss, take_profit, indicators)
                    if asyncio.iscoroutinefunction(place_order):
                        order = await place_order(*place_args)
                    else:
                        order = await asyncio.to_thread(place_order, *place_args)

                    # Ensure order is a dictionary
                    if not isinstance(order, dict):
                        raise ValueError(f"Unexpected order response type: {type(order)}")

                    # Handle exchange-specific order response
                    if self.exchange_name == "binance":
                        if "error" not in order and order.get("status", "").lower() in ["new", "filled"]:
                            order_id = str(order.get("order_id", ""))
                            status = order.get("status", "pending").lower()
                            self.db.update_trade(int(trade.id), {"order_id": order_id, "status": status})  # type: ignore
                            self.db.session.commit()
                            self.trade_count += 1
                            self.successful_trades += 1
                            logger.info(f"Trade executed: {symbol} {side} {qty} @ {entry_price}, order_id={order_id}")
                            return True
                        else:
                            error_msg = order.get("error", "Unknown error")
                            logger.error(f"Trade execution failed for {symbol}: {error_msg}")
                            self.trade_count += 1
                            self.failed_trades += 1
                            self.db.update_trade(int(trade.id), {"status": "failed", "error": error_msg})  # type: ignore
                            self.db.session.commit()
                            return False
                    elif self.exchange_name == "bybit":
                        if to_float(order.get("retCode", -1)) == 0:
                            order_id = order.get("result", {}).get("orderId", "")
                            status = "filled" if order.get("result", {}).get("orderStatus", "") == "Filled" else "pending"
                            self.db.update_trade(int(trade.id), {"order_id": order_id, "status": status})  # type: ignore
                            self.db.session.commit()
                            self.trade_count += 1
                            self.successful_trades += 1
                            logger.info(f"Trade executed: {symbol} {side} {qty} @ {entry_price}, order_id={order_id}")
                            return True
                        else:
                            error_msg = order.get("retMsg", "Unknown error")
                            logger.error(f"Trade execution failed for {symbol}: {error_msg}")
                            self.trade_count += 1
                            self.failed_trades += 1
                            self.db.update_trade(int(trade.id), {"status": "failed", "error": error_msg})  # type: ignore
                            self.db.session.commit()
                            return False

                except (APIException, APIRateLimitException, APIAuthenticationException, APITimeoutException) as e:
                    self._consecutive_failures += 1
                    if attempt < max_retries - 1:
                        retry_delay = 2 ** attempt  # Exponential backoff
                        logger.warning(f"API error on attempt {attempt + 1}, retrying in {retry_delay}s: {str(e)}")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        self.trade_count += 1
                        self.failed_trades += 1
                        # best-effort update
                        try:
                            self.db.update_trade(int(trade.id), {"status": "failed", "error": str(e)})  # type: ignore
                            self.db.session.commit()
                        except Exception:
                            pass
                        logger.error(f"Max retries exceeded for {symbol}: {str(e)}")
                        return False
                except Exception as e:
                    self.trade_count += 1
                    self.failed_trades += 1
                    try:
                        self.db.update_trade(int(trade.id), {"status": "failed", "error": str(e)})  # type: ignore
                        self.db.session.commit()
                    except Exception:
                        pass
                    logger.error(f"Unexpected error executing trade for {symbol}: {e}")
                    return False

        except Exception as e:
            self.trade_count += 1
            self.failed_trades += 1
            if 'trade' in locals():
                try:
                    self.db.update_trade(int(trade.id), {"status": "failed", "error": str(e)})  # type: ignore
                    self.db.session.commit()
                except Exception:
                    pass
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False
        finally:
            try:
                self.db.session.close()
            except Exception:
                pass

        # Ensure a boolean return in all code paths (static analyzers want this)
        return False

    def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        try:
            if TRADING_MODE == "virtual":
                # This is a simplified representation for virtual mode.
                # In a real scenario, this would fetch from a virtual balance store.
                trades = self.db.get_trades(exchange=self.exchange_name, status="open") or []
                total_pnl = sum(to_float(getattr(t, "pnl", 0)) for t in trades)
                initial_balance = to_float(self.settings.get("VIRTUAL_BALANCE", 10000.0))
                return {
                    "USDT": {
                        "total": initial_balance + total_pnl,
                        "free": initial_balance + total_pnl,
                        "used": 0,
                        "currency": "USDT"
                    }
                }
            balance = self.client.get_account_balance() or {}
            formatted_balance = {
                currency: {
                    "total": to_float(getattr(b, "total", 0)),
                    "free": to_float(getattr(b, "available", getattr(b, "free", 0))),
                    "used": to_float(getattr(b, "used", 0)),
                    "currency": currency
                } for currency, b in (balance.items() if isinstance(balance, dict) else [])
            }
            return formatted_balance
        except Exception as e:
            logger.error(f"Error fetching real account balance: {e}")
            return {'USDT': {'total': 0.0, 'free': 0.0, 'used': 0.0, 'currency': 'USDT'}}

    def execute_virtual_trade(self, signal: Dict[str, Any]) -> bool:
        """Execute a virtual trade based on signal"""
        success = False
        try:
            symbol = signal.get("symbol", '').replace('/', '')
            side = signal.get("side", "BUY")
            entry_price = to_float(signal.get("entry", 0))

            if not symbol or entry_price <= 0:
                logger.error(f"Invalid signal data: {signal}")
                return False

            # Calculate position size
            qty = self.calculate_position_size(symbol, entry_price)

            # Create virtual trade
            trade = Trade(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=entry_price,
                status="open",
                virtual=True,
                leverage=signal.get("leverage", 10),
                sl=signal.get("sl"),
                tp=signal.get("tp"),
                trail=signal.get("trail"),
                margin_usdt=signal.get("margin_usdt"),
                exchange=self.exchange_name,
                signal_id=signal.get("id"),
                indicators=signal.get("indicators", {})
            )

            # Add to database
            success = bool(self.db.add_trade(trade))
            if success:
                logger.info(f"Virtual trade executed: {symbol} {side} {qty} @ {entry_price}")

            return success

        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return False

    def close_virtual_trade(self, trade_id: int, exit_price: float) -> bool:
        """Close a virtual trade"""
        try:
            trade = self.db.get_trade_by_id(trade_id)
            if not trade or not is_true(getattr(trade, "virtual", False)):
                logger.error(f"Virtual trade {trade_id} not found or not virtual")
                return False

            # Calculate PnL
            pnl = self.calculate_virtual_pnl({
                "symbol": getattr(trade, "symbol", ""),
                "entry_price": getattr(trade, "entry_price", 0),
                "qty": getattr(trade, "qty", 0),
                "side": getattr(trade, "side", "BUY")
            }, exit_price) # Pass exit_price here

            # Update trade
            updates = {
                "status": "closed",
                "exit_price": exit_price,
                "pnl": pnl
            }

            success = bool(self.db.update_trade(trade_id, updates))
            if success:
                logger.info(f"Virtual trade closed: {trade.symbol} PnL: {pnl}")

            return success

        except Exception as e:
            logger.error(f"Error closing virtual trade: {e}")
            return False

    def calculate_virtual_pnl(self, trade: Dict[str, Any], exit_price: Optional[float] = None) -> float:
        """Calculate PnL for a virtual trade"""
        try:
            symbol = trade.get("symbol", '').replace('/', '')
            entry_price = to_float(trade.get("entry_price", 0))
            qty = to_float(trade.get("qty", 0))
            side = (trade.get("side", "BUY") or "BUY").lower()

            # Use provided exit_price if available, otherwise fetch current price
            current_price = to_float(exit_price) if exit_price is not None else to_float(self.client.get_current_price(symbol))

            if entry_price <= 0 or current_price <= 0 or qty <= 0:
                return 0.0

            if side == "buy":
                return (current_price - entry_price) * qty
            else:
                return (entry_price - current_price) * qty
        except Exception as e:
            logger.error(f"Error calculating virtual PnL for {trade.get('symbol')}: {e}")
            return 0.0

    def _update_capital_json(self, account_type: str, amount: float):
        """Update capital.json file with trade amounts"""
        try:
            import json
            capital_file = "capital.json"

            # Load existing capital data
            if os.path.exists(capital_file):
                with open(capital_file, 'r') as f:
                    capital_data = json.load(f)
            else:
                # Default structure if capital.json doesn't exist
                capital_data = {
                    "real": {"capital": 0.0, "available": 0.0, "used": 0.0, "start_balance": 0.0, "currency": "USDT"},
                    "virtual": {"capital": 100.0, "available": 100.0, "used": 0.0, "start_balance": 100.0, "currency": "USDT"}
                }

            # Update the appropriate account
            if account_type in capital_data:
                # For trades, 'amount' typically represents the value of the position being opened.
                # We add this to 'used' and subtract from 'available'.
                capital_data[account_type]["used"] = to_float(capital_data[account_type]["used"]) + abs(amount)
                capital_data[account_type]["available"] = to_float(capital_data[account_type]["available"]) - abs(amount)
                
                # Ensure available and used don't go below zero due to floating point inaccuracies or order of operations
                capital_data[account_type]["available"] = max(0.0, capital_data[account_type]["available"])
                capital_data[account_type]["used"] = max(0.0, capital_data[account_type]["used"])

                # Save back to file
                with open(capital_file, 'w') as f:
                    json.dump(capital_data, f, indent=4)

                # Also update database wallet balance
                self.db.update_wallet_balance(
                    account_type,
                    available=capital_data[account_type]["available"],
                    used=capital_data[account_type]["used"],
                    exchange=self.exchange_name
                )

                logger.info(f"Updated {account_type} capital: available={capital_data[account_type]['available']}, used={capital_data[account_type]['used']}")
            else:
                logger.error(f"Account type '{account_type}' not found in capital data.")
        except Exception as e:
            logger.error(f"Error updating capital.json: {e}")