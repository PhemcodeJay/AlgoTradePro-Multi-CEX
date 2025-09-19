import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from db import db_manager, Trade, WalletBalance
from settings import load_settings
from logging_config import get_trading_logger
from exceptions import TradingException, create_error_context

logger = get_trading_logger('engine')


class TradingEngine:
    def __init__(self):
        try:
            # Load base + exchange-specific settings
            base_settings = load_settings()
            self.exchange_name = base_settings.get("EXCHANGE", "binance").lower()
            exchange_overrides = base_settings.get(self.exchange_name, {})

            # Set exchange attribute (Pylance-friendly)
            self.exchange = getattr(self, "exchange", self.exchange_name)

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
            self._candle_cache = {}

            # Position safety limits
            self.max_position_size = self.settings.get("MAX_POSITION_SIZE", 10000.0)
            self.max_open_positions = self.settings.get("MAX_OPEN_POSITIONS", 10)
            self.max_daily_loss = self.settings.get("MAX_DAILY_LOSS", 1000.0)
            self.max_risk_per_trade = self.settings.get("MAX_RISK_PER_TRADE", 0.05)

            # Trading state management
            self._trading_enabled = True
            self._emergency_stop = False
            self._last_health_check = None
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
            logger.error(f"Failed to initialize trading engine: {str(e)}")
            raise TradingException(
                f"Trading engine initialization failed: {str(e)}",
                context=error_context,
                original_exception=e
            )

    def reload_settings(self) -> None:
        """Reload settings (used when user switches exchange in UI)."""
        try:
            base_settings = load_settings()
            self.exchange_name = base_settings.get("EXCHANGE", "binance").lower()
            exchange_overrides = base_settings.get(self.exchange_name, {})
            self.settings = {**base_settings, **exchange_overrides}

            logger.info(f"Settings reloaded for {self.exchange_name}")
        except Exception as e:
            logger.error(f"Failed to reload settings: {e}")

    def switch_exchange(self, new_exchange: str) -> bool:
        """Switch exchange dynamically (binance/bybit)."""
        try:
            self.settings["EXCHANGE"] = new_exchange.lower()
            self.reload_settings()

            if self.exchange_name == "binance":
                from binance_client import BinanceClient
                self.client = BinanceClient()
            elif self.exchange_name == "bybit":
                from bybit_client import BybitClient
                self.client = BybitClient()
            else:
                raise TradingException(f"Unsupported exchange: {new_exchange}")

            logger.info(f"Switched trading engine to {new_exchange}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch exchange: {e}")
            return False

    def _reset_daily_stats(self):
        try:
            current_time = datetime.now(timezone.utc)
            if current_time >= self._daily_reset_time + timedelta(days=1):
                self._daily_pnl = 0.0
                self._daily_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                logger.info("Daily trading statistics reset")
        except Exception as e:
            logger.warning(f"Failed to reset daily stats: {str(e)}")

    def _check_emergency_conditions(self) -> bool:
        try:
            self._reset_daily_stats()
            if self._daily_pnl <= -self.max_daily_loss:
                logger.critical(f"Daily loss limit exceeded: {self._daily_pnl} <= -{self.max_daily_loss}")
                self._emergency_stop = True
                return False
            if self._consecutive_failures >= 10:
                logger.critical(f"Too many consecutive failures: {self._consecutive_failures}")
                self._emergency_stop = True
                return False
            api_health = self.client.get_connection_health()
            if api_health['status'] not in ['healthy', 'degraded']:
                logger.warning(f"API health check failed: {api_health['status']}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {str(e)}")
            return False

    def is_trading_enabled(self) -> bool:
        return self._trading_enabled and not self._emergency_stop and self._check_emergency_conditions()

    def enable_trading(self) -> bool:
        try:
            if self._emergency_stop:
                logger.warning("Cannot enable trading: Emergency stop is active")
                return False
            if not self._check_emergency_conditions():
                logger.warning("Cannot enable trading: Emergency conditions detected")
                return False
            self._trading_enabled = True
            logger.info("Trading enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable trading: {str(e)}")
            return False

    def disable_trading(self, reason: str = "Manual disable") -> bool:
        try:
            self._trading_enabled = False
            logger.warning(f"Trading disabled: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to disable trading: {str(e)}")
            return False

    def emergency_stop(self, reason: str = "Emergency stop triggered") -> bool:
        try:
            self._emergency_stop = True
            self._trading_enabled = False
            logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            return True
        except Exception as e:
            logger.critical(f"Failed to activate emergency stop: {str(e)}")
            return False

    def get_settings(self) -> Tuple[int, int]:
        return self.settings.get("SCAN_INTERVAL", 3600), self.settings.get("TOP_N_SIGNALS", 5)

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        try:
            self.settings.update(new_settings)
            with open("settings.json", "w") as f:
                json.dump(self.settings, f, indent=2)
            logger.info("Settings updated")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False

    def get_cached_candles(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        try:
            cache_key = f"{symbol}_{interval}_{limit}"
            now = datetime.now()
            if cache_key in self._candle_cache:
                cached_time, cached_data = self._candle_cache[cache_key]
                if (now - cached_time).total_seconds() < 300:
                    return cached_data
            candles = self.client.get_klines(symbol, interval, limit)
            if candles:
                self._candle_cache[cache_key] = (now, candles)
                return candles
            return []
        except Exception as e:
            logger.error(f"Error getting candles for {symbol}: {e}")
            return []

    def get_usdt_symbols(self) -> List[str]:
        return self.settings.get("SYMBOLS", ["BTCUSDT", "ETHUSDT", "DOGEUSDT"])

    def calculate_position_size(self, symbol: str, entry_price: float,
                              risk_percent: Optional[float] = None, leverage: Optional[int] = None) -> float:
        try:
            risk_pct = risk_percent or self.settings.get("RISK_PCT", 0.01)
            lev = leverage or self.settings.get("LEVERAGE", 10)
            wallet_balance = self.db.get_wallet_balance("virtual")
            if not wallet_balance:
                self.db.migrate_capital_json_to_db()
                wallet_balance = self.db.get_wallet_balance("virtual")
            balance = wallet_balance.available if wallet_balance else 100.0
            risk_amount = balance * risk_pct
            position_value = risk_amount * lev
            position_size = position_value / entry_price
            return round(position_size, 6)
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01

    def calculate_virtual_pnl(self, trade: Dict) -> float:
        try:
            current_price = self.client.get_current_price(trade["symbol"])
            entry_price = float(trade.get("entry_price", 0))
            qty = float(trade.get("qty", 0))
            side = trade.get("side", "Buy").upper()
            if current_price <= 0 or entry_price <= 0:
                return 0.0
            if side in ["BUY", "LONG"]:
                pnl = (current_price - entry_price) * qty
            else:
                pnl = (entry_price - current_price) * qty
            return round(pnl, 2)
        except Exception as e:
            logger.error(f"Error calculating virtual PnL: {e}")
            return 0.0

    def get_open_virtual_trades(self) -> List[Trade]:
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if trade.virtual and trade.status == "open"]
        except Exception as e:
            logger.error(f"Error getting open virtual trades: {e}")
            return []

    def get_open_real_trades(self) -> List[Trade]:
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if not trade.virtual and trade.status == "open"]
        except Exception as e:
            logger.error(f"Error getting open real trades: {e}")
            return []

    def get_closed_virtual_trades(self) -> List[Trade]:
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if trade.virtual and trade.status == "closed"]
        except Exception as e:
            logger.error(f"Error getting closed virtual trades: {e}")
            return []

    def get_closed_real_trades(self) -> List[Trade]:
        try:
            all_trades = self.db.get_trades()
            return [trade for trade in all_trades if not trade.virtual and trade.status == "closed"]
        except Exception as e:
            logger.error(f"Error getting closed real trades: {e}")
            return []

    def get_trade_statistics(self) -> Dict[str, Any]:
        try:
            all_trades = self.db.get_trades()
            virtual_trades = [t for t in all_trades if t.virtual]
            real_trades = [t for t in all_trades if not t.virtual]

            def calc_stats(trades):
                if not trades:
                    return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0, "profitable_trades": 0}
                pnls = [t.pnl or 0 for t in trades]
                profitable = len([p for p in pnls if p > 0])
                return {
                    "total_trades": len(trades),
                    "win_rate": (profitable / len(trades)) * 100,
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls),
                    "profitable_trades": profitable
                }

            virtual_stats = calc_stats(virtual_trades)
            real_stats = calc_stats(real_trades)
            overall_stats = calc_stats(all_trades)

            return {
                **overall_stats,
                "virtual_total_trades": virtual_stats["total_trades"],
                "virtual_win_rate": virtual_stats["win_rate"],
                "virtual_total_pnl": virtual_stats["total_pnl"],
                "real_total_trades": real_stats["total_trades"],
                "real_win_rate": real_stats["win_rate"],
                "real_total_pnl": real_stats["total_pnl"]
            }
        except Exception as e:
            logger.error(f"Error calculating trade statistics: {e}")
            return {}

    def update_virtual_balances(self, pnl: float, mode: str = "virtual"):
        try:
            wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                default_balance = WalletBalance(
                    trading_mode=mode,
                    capital=1000.0 if mode == "virtual" else 0.0,
                    available=1000.0 if mode == "virtual" else 0.0,
                    used=0.0,
                    start_balance=1000.0 if mode == "virtual" else 0.0,
                    currency="USDT",
                    updated_at=datetime.utcnow(),
                )
                self.db.update_wallet_balance(default_balance)
                wallet_balance = self.db.get_wallet_balance(mode)
            if not wallet_balance:
                logger.error(f"Failed to get or create wallet balance for mode: {mode}")
                return
            new_available = max(0.0, wallet_balance.available + pnl)
            new_capital = wallet_balance.capital + pnl
            new_used = max(0.0, new_capital - new_available)
            updated_balance = WalletBalance(
                trading_mode=mode,
                capital=new_capital,
                available=new_available,
                used=new_used,
                start_balance=wallet_balance.start_balance,
                currency=wallet_balance.currency,
                updated_at=datetime.utcnow(),
                id=wallet_balance.id,
            )
            self.db.update_wallet_balance(updated_balance)
            logger.info(f"Updated {mode} balance: PnL {pnl:+.2f} -> available {new_available:.2f}")
        except Exception as e:
            logger.error(f"Error updating virtual balances: {e}")

    def sync_real_balance(self):
        try:
            if not hasattr(self, 'client') or self.client is None:
                logger.error("Exchange client not initialized. Check API credentials in .env")
                return False
            if not self.client.is_connected():
                logger.warning("Client not connected. Attempting to reconnect...")
                try:
                    self.switch_exchange(self.exchange_name)
                    if not self.client.is_connected():
                        logger.error("Reconnection failed.")
                        return False
                    logger.info("Client reconnected successfully")
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}", exc_info=True)
                    return False
            result = self.client._make_request("GET", "/api/v3/account", {})
            if not result or "balances" not in result:
                logger.warning("No account data in exchange response")
                return False
            usdt_balance = next((asset for asset in result["balances"] if asset.get("asset") == "USDT"), None)
            if not usdt_balance:
                logger.warning("No USDT balance found in response")
                return False
            total_available = float(usdt_balance.get("free", "0") or 0)
            used = float(usdt_balance.get("locked", "0") or 0)
            total_equity = total_available + used
            if total_available == 0:
                logger.warning("Total available balance is 0")
            existing_balance: Optional[WalletBalance] = self.db.get_wallet_balance("real")
            start_balance = existing_balance.start_balance if existing_balance and existing_balance.start_balance > 0 else total_equity
            wallet_balance = WalletBalance(
                trading_mode="real",
                capital=total_equity,
                available=total_available,
                used=used,
                start_balance=start_balance,
                currency="USDT",
                updated_at=datetime.now(timezone.utc),
                id=existing_balance.id if existing_balance else None,
            )
            success = self.db.update_wallet_balance(wallet_balance)
            if success:
                logger.info(f"✅ Real balance synced: Capital=${total_equity:.2f}, Available=${total_available:.2f}, Used=${used:.2f}")
                return True
            else:
                logger.error("Failed to update wallet balance in database")
                return False
        except Exception as e:
            logger.error(f"❌ Error syncing real balance: {e}", exc_info=True)
            return False

    def execute_virtual_trade(self, signal: Dict, trading_mode: str = "virtual") -> bool:
        try:
            symbol = signal.get("symbol")
            if not symbol:
                logger.error("Symbol is required for executing trade")
                return False
            side = signal.get("side", "Buy")
            entry_price = signal.get("entry") or self.client.get_current_price(symbol)
            if entry_price <= 0:
                logger.error(f"Invalid entry price for {symbol}")
                return False
            position_size = self.calculate_position_size(symbol, entry_price)
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": position_size,
                "entry_price": entry_price,
                "order_id": f"virtual_{symbol}_{int(datetime.now().timestamp())}",
                "virtual": trading_mode == "virtual",
                "status": "open",
                "score": signal.get("score"),
                "strategy": signal.get("strategy", "Auto"),
                "leverage": signal.get("leverage", 10)
            }
            success = self.db.add_trade(trade_data)
            if success:
                logger.info(f"Virtual trade executed: {symbol} {side} @ {entry_price}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error executing virtual trade: {e}")
            return False

    def close(self):
        try:
            if hasattr(self, "client"):
                self.client.close()
            logger.info("Trading engine closed")
        except Exception as e:
            logger.error(f"Error closing trading engine: {e}")
