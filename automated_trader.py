# automated_trader.py
import asyncio
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from signal_generator import generate_signals, get_top_symbols
from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_trading_logger

logger = get_trading_logger('automated')


class AutomatedTrader:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_scan_time: Optional[datetime] = None
        self.scan_count = 0
        self.total_signals_generated = 0
        self.successful_trades = 0
        self.failed_trades = 0

        # defaults
        self.scan_interval = 3600          # seconds
        self.max_symbols = 50
        self.top_n_signals = 10
        self.timeframe = "1h"
        self.auto_trading_enabled = False
        self.notification_enabled = True

        self._load_settings()
        logger.info(
            "AutomatedTrader initialized | Exchange: %s | Account: %s",
            self.trading_engine.exchange,
            self.trading_engine.account_type,
        )

    # ------------------------------------------------------------------ #
    #  SETTINGS
    # ------------------------------------------------------------------ #
    def _load_settings(self):
        """Load settings from TradingEngine safely."""
        try:
            settings = self.trading_engine.get_settings() or {}

            self.scan_interval = int(settings.get("SCAN_INTERVAL", self.scan_interval))
            self.max_symbols = int(settings.get("MAX_SYMBOLS", self.max_symbols))
            self.top_n_signals = int(settings.get("TOP_N_SIGNALS", self.top_n_signals))
            self.timeframe = str(settings.get("TIMEFRAME", self.timeframe))
            self.auto_trading_enabled = bool(settings.get("AUTO_TRADING_ENABLED", self.auto_trading_enabled))
            self.notification_enabled = bool(settings.get("NOTIFICATION_ENABLED", self.notification_enabled))

            logger.info(
                "Settings: interval=%ss, max_symbols=%s, top_n=%s, tf=%s, auto=%s, notif=%s",
                self.scan_interval, self.max_symbols, self.top_n_signals,
                self.timeframe, self.auto_trading_enabled, self.notification_enabled
            )
        except Exception as exc:
            logger.error("Error loading settings: %s", exc)

    # ------------------------------------------------------------------ #
    #  START / STOP
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        if self.running:
            logger.warning("Automated trader already running")
            return False

        if not self.trading_engine.is_trading_enabled():
            logger.warning("Trading engine disabled")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        logger.info("Automated trader started")
        return True

    def stop(self) -> bool:
        if not self.running:
            logger.warning("Automated trader not running")
            return False

        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=15)
        logger.info("Automated trader stopped")
        return True

    # ------------------------------------------------------------------ #
    #  MAIN LOOP
    # ------------------------------------------------------------------ #
    async def _trading_loop(self):
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.running:
            try:
                # Original sleep was here, replaced by the backoff mechanism below.
                # If no error, this will effectively be a 1-second sleep, then cycle.
                # A more precise sleep for the scan_interval would be:
                # await asyncio.sleep(self.scan_interval)
                # However, the original code had a loop for sleep, so we maintain that logic.
                # For a production system, a single asyncio.sleep(self.scan_interval) is better.

                await self._execute_trading_cycle() # Execute the cycle first

                # Sleep with interruption support, similar to original code's intent
                for _ in range(self.scan_interval):
                    if not self.running:
                        break
                    await asyncio.sleep(1) # Use asyncio.sleep for cooperative multitasking

                consecutive_errors = 0  # Reset on success
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Trading loop error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}", exc_info=True)

                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("Max consecutive errors reached - stopping trading loop")
                    self.stop()
                    break

                # Exponential backoff
                backoff = min(300, 60 * (2 ** consecutive_errors))
                logger.warning(f"Backing off for {backoff}s before retry")
                await asyncio.sleep(backoff)

    # ------------------------------------------------------------------ #
    #  ONE CYCLE
    # ------------------------------------------------------------------ #
    async def _execute_trading_cycle(self):
        try:
            self.scan_count += 1
            self.last_scan_time = datetime.now(timezone.utc)
            exchange = self.trading_engine.exchange

            logger.info("Starting scan #%s on %s (%s mode) for user %s", 
                       self.scan_count, exchange.upper(), 
                       self.trading_engine.account_type,
                       self.trading_engine.user_id)

            # 1. Get top symbols
            symbols = get_top_symbols(exchange, limit=self.max_symbols)
            if not symbols:
                logger.warning("No symbols fetched for %s", exchange)
                return

            # 2. Generate signals with proper error handling
            signals = [] # Initialize signals to empty list
            try:
                raw_signals = generate_signals(
                    exchange=exchange,
                    timeframe=self.timeframe,
                    max_symbols=self.max_symbols,
                    user_id=self.trading_engine.user_id
                )

                # Ensure signals have proper structure
                if raw_signals:
                    for sig in raw_signals:
                        if 'exchange' not in sig:
                            sig['exchange'] = exchange
                        if 'user_id' not in sig:
                            sig['user_id'] = self.trading_engine.user_id
                        # Ensure TP/SL are included
                        if 'sl' not in sig:
                            sig['sl'] = None
                        if 'tp' not in sig:
                            sig['tp'] = None
                        signals.append(sig) # Append validated signal

            except Exception as sig_err:
                logger.error("Signal generation failed for %s: %s", exchange, sig_err)
                # Continue execution if signal generation fails, to allow other parts of the cycle to run
                # If signals is empty, the subsequent logic will handle it.

            self.total_signals_generated += len(signals)

            if not signals:
                logger.info("No high-confidence signals this cycle")
                return

            logger.info("Generated %s signals for %s", len(signals), exchange)

            # 3. Send notifications
            if self.notification_enabled:
                try:
                    send_all_notifications(signals)
                except Exception as e:
                    logger.error("Notification failed: %s", e)

            # 4. Execute trades (only if auto-trading enabled)
            if self.auto_trading_enabled:
                open_positions = len(self.trading_engine.get_open_trades())
                max_positions = self.trading_engine.max_open_positions

                for sig in signals:
                    if open_positions >= max_positions:
                        logger.info("Max open positions reached (%s)", max_positions)
                        break

                    symbol = sig["symbol"]
                    side = sig["side"]

                    try:
                        logger.info("Executing %s %s @ %.2f on %s (%s)", 
                                   side, symbol, sig["entry"], exchange,
                                   self.trading_engine.account_type)

                        # Execute trade with proper await
                        success = await self._execute_trade_async(sig)

                        if success:
                            self.successful_trades += 1
                            open_positions += 1
                            logger.info("Trade executed: %s %s on %s", side, symbol, exchange)
                        else:
                            self.failed_trades += 1
                            logger.warning("Trade failed: %s %s on %s", side, symbol, exchange)
                    except Exception as exc:
                        self.failed_trades += 1
                        logger.error("Execution error for %s on %s: %s", symbol, exchange, exc, exc_info=True)

            # 5. Update ML model
            self._update_ml_feedback()

        except Exception as exc:
            logger.error("Error in trading cycle: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    #  ML FEEDBACK
    # ------------------------------------------------------------------ #
    def _update_ml_feedback(self):
        try:
            trades = self.trading_engine.db.get_trades(
                hours=24,
                exchange=self.trading_engine.exchange,
                user_id=self.trading_engine.user_id,
                virtual=(self.trading_engine.account_type == "virtual")
            )
            if not trades:
                return

            ml = MLFilter(user_id=self.trading_engine.user_id, exchange=self.trading_engine.exchange)
            updated = 0
            for tr in trades:
                if tr.pnl is None:
                    continue
                try:
                    signal = {
                        "symbol": tr.symbol,
                        "indicators": tr.indicators or {},
                        "pnl": float(tr.pnl),
                        "side": tr.side,
                        "entry": tr.entry_price,
                    }
                    outcome = float(tr.pnl) > 0
                    ml.update_model_with_feedback(signal, outcome)
                    updated += 1
                except Exception as e:
                    logger.debug("ML feedback skip for trade %s: %s", tr.id, e)

            if updated > 0:
                logger.info("ML model updated with %s trades", updated)
        except Exception as exc:
            logger.error("ML feedback failed: %s", exc)

    # ------------------------------------------------------------------ #
    #  STATUS
    # ------------------------------------------------------------------ #
    def get_status(self) -> Dict[str, Any]:
        bal = self.trading_engine.get_balance()
        stats = self.trading_engine.get_trade_statistics(mode=self.trading_engine.account_type)

        return {
            "running": self.running,
            "exchange": self.trading_engine.exchange,
            "account_type": self.trading_engine.account_type,
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "scan_count": self.scan_count,
            "total_signals_generated": self.total_signals_generated,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "scan_interval": self.scan_interval,
            "max_symbols": self.max_symbols,
            "top_n_signals": self.top_n_signals,
            "timeframe": self.timeframe,
            "auto_trading_enabled": self.auto_trading_enabled,
            "notification_enabled": self.notification_enabled,
            "open_positions": len(self.trading_engine.get_open_trades()),
            "max_open_positions": self.trading_engine.max_open_positions,
            "balance_total": round(bal["total"], 2),
            "balance_available": round(bal["available"], 2),
            "daily_pnl": round(self.trading_engine._daily_pnl, 2),
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"],
        }

    # ------------------------------------------------------------------ #
    #  MANUAL SCAN
    # ------------------------------------------------------------------ #
    def force_scan(self) -> Dict[str, Any]:
        if not self.trading_engine.is_trading_enabled():
            return {"success": False, "error": "Trading engine disabled"}

        logger.info("Force scan triggered")
        try:
            # Use asyncio.run for synchronous context like this force_scan method
            asyncio.run(self._execute_trading_cycle())
            # Re-run generate_signals to get the count, as _execute_trading_cycle might have modified state
            # Or, better, capture the count directly from _execute_trading_cycle if possible.
            # For now, re-running to match original logic.
            signals_generated_count = len(generate_signals(
                exchange=self.trading_engine.exchange,
                timeframe=self.timeframe,
                max_symbols=self.max_symbols,
                user_id=self.trading_engine.user_id
            ))
            return {
                "success": True,
                "message": "Force scan completed",
                "scan_time": datetime.now(timezone.utc).isoformat(),
                "signals_generated": signals_generated_count
            }
        except Exception as exc:
            logger.error("Force scan failed: %s", exc)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    #  SETTINGS UPDATE
    # ------------------------------------------------------------------ #
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        try:
            updated = False
            if "scan_interval" in new_settings:
                self.scan_interval = max(60, int(new_settings["scan_interval"]))
                updated = True
            if "max_symbols" in new_settings:
                self.max_symbols = max(10, int(new_settings["max_symbols"]))
                updated = True
            if "top_n_signals" in new_settings:
                self.top_n_signals = max(1, int(new_settings["top_n_signals"]))
                updated = True
            if "timeframe" in new_settings:
                self.timeframe = str(new_settings["timeframe"])
                updated = True
            if "auto_trading_enabled" in new_settings:
                self.auto_trading_enabled = bool(new_settings["auto_trading_enabled"])
                updated = True
            if "notification_enabled" in new_settings:
                self.notification_enabled = bool(new_settings["notification_enabled"])
                updated = True

            if updated:
                logger.info("AutomatedTrader settings updated: %s", new_settings)
            return updated
        except Exception as exc:
            logger.error("Failed to update settings: %s", exc)
            return False

    # Helper method to execute trades asynchronously
    async def _execute_trade_async(self, signal: Dict[str, Any]) -> bool:
        try:
            # Ensure we're in an async context
            if asyncio.get_event_loop().is_running():
                result = await self.trading_engine.execute_trade(signal)
            else:
                result = asyncio.run(self.trading_engine.execute_trade(signal))
            return result
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}", exc_info=True)
            return False