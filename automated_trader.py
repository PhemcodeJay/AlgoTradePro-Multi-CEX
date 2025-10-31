# automated_trader.py
import asyncio
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List

from signal_generator import generate_signals, get_top_symbols
from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_trading_logger

logger = get_trading_logger('automated')


class AutomatedTrader:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_scan_time: datetime | None = None
        self.scan_count = 0
        self.total_signals_generated = 0
        self.successful_trades = 0
        self.failed_trades = 0

        # defaults – will be overwritten by _load_settings()
        self.scan_interval = 3600          # seconds
        self.top_n_signals = 10
        self.auto_trading_enabled = False
        self.notification_enabled = True

        self._load_settings()
        logger.info(
            "AutomatedTrader initialized for %s",
            self.trading_engine.exchange_name or self.trading_engine.exchange,
        )

    # ------------------------------------------------------------------ #
    #  SETTINGS
    # ------------------------------------------------------------------ #
    def _load_settings(self):
        """Load settings from the engine – safe, no unpacking errors."""
        try:
            settings = self.trading_engine.get_settings() or {}

            # ---- interval & top-N ------------------------------------------------
            self.scan_interval = int(settings.get("SCAN_INTERVAL", self.scan_interval))
            self.top_n_signals = int(settings.get("TOP_N_SIGNALS", self.top_n_signals))

            # ---- feature toggles -------------------------------------------------
            self.auto_trading_enabled = bool(
                settings.get("AUTO_TRADING_ENABLED", self.auto_trading_enabled)
            )
            self.notification_enabled = bool(
                settings.get("NOTIFICATION_ENABLED", self.notification_enabled)
            )

            logger.info(
                "Settings loaded – interval=%ss, top_n=%s, auto=%s, notif=%s",
                self.scan_interval,
                self.top_n_signals,
                self.auto_trading_enabled,
                self.notification_enabled,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Error loading settings: %s", exc)

    # ------------------------------------------------------------------ #
    #  START / STOP
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        if self.running:
            logger.warning("Automated trader is already running")
            return False
        if not self.trading_engine.is_trading_enabled():
            logger.warning("Trading engine is not enabled")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.thread.start()
        logger.info("Automated trader started")
        return True

    def stop(self) -> bool:
        if not self.running:
            logger.warning("Automated trader is not running")
            return False

        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        logger.info("Automated trader stopped")
        return True

    # ------------------------------------------------------------------ #
    #  MAIN LOOP
    # ------------------------------------------------------------------ #
    def _trading_loop(self):
        logger.info("Automated trading loop started")
        while self.running:
            try:
                if not self.trading_engine.is_trading_enabled():
                    logger.warning("Trading engine disabled – stopping loop")
                    self.stop()
                    return

                asyncio.run(self._execute_trading_cycle())
                time.sleep(self.scan_interval)
            except Exception as exc:  # pragma: no cover
                logger.error("Error in trading loop: %s", exc)
                time.sleep(60)

    # ------------------------------------------------------------------ #
    #  ONE CYCLE
    # ------------------------------------------------------------------ #
    async def _execute_trading_cycle(self):
        try:
            self.scan_count += 1
            self.last_scan_time = datetime.now(timezone.utc)

            exchange = self.trading_engine.exchange_name or self.trading_engine.exchange
            symbols = get_top_symbols(exchange, limit=50)
            if not symbols:
                logger.warning("No symbols retrieved for %s", exchange)
                return

            signals = generate_signals(
                exchange,
                symbols,
                interval="60",
                top_n=self.top_n_signals,
                user_id=self.trading_engine.user_id,
            )
            self.total_signals_generated += len(signals)

            if not signals:
                logger.info("No valid signals generated this cycle")
                return

            # ---- notifications -------------------------------------------------
            if self.notification_enabled:
                for sig in signals:
                    send_all_notifications(sig)

            # ---- auto-execution ------------------------------------------------
            if self.auto_trading_enabled:
                for sig in signals:
                    try:
                        symbol = sig.get("symbol", "").replace("/", "")
                        sig["symbol"] = symbol
                        success = await self.trading_engine.execute_trade(sig)
                        if success:
                            self.successful_trades += 1
                            logger.info("Trade executed for %s", symbol)
                        else:
                            self.failed_trades += 1
                            logger.warning("Trade execution failed for %s", symbol)
                    except Exception as exc:  # pragma: no cover
                        self.failed_trades += 1
                        logger.error(
                            "Error executing trade for %s: %s",
                            sig.get("symbol"),
                            exc,
                        )

            self._update_ml_feedback()
        except Exception as exc:  # pragma: no cover
            logger.error("Error in trading cycle: %s", exc)

    # ------------------------------------------------------------------ #
    #  ML FEEDBACK
    # ------------------------------------------------------------------ #
    def _update_ml_feedback(self):
        try:
            trades = self.trading_engine.db.get_trades(
                hours=24,
                exchange=self.trading_engine.exchange_name or self.trading_engine.exchange,
                user_id=self.trading_engine.user_id,
            )
            if not trades:
                return

            ml = MLFilter()
            for tr in trades:
                try:
                    signal = {
                        "symbol": tr.symbol,
                        "indicators": tr.indicators or {},
                        "pnl": tr.pnl or 0,
                    }
                    outcome = (tr.pnl or 0) > 0
                    ml.update_model_with_feedback(signal, outcome)
                except Exception as exc:  # pragma: no cover
                    logger.error("ML feedback error for trade %s: %s", tr.id, exc)

            logger.info("ML model updated with %s recent trades", len(trades))
        except Exception as exc:  # pragma: no cover
            logger.error("Error updating ML model: %s", exc)

    # ------------------------------------------------------------------ #
    #  STATUS
    # ------------------------------------------------------------------ #
    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "last_scan_time": (
                self.last_scan_time.isoformat() if self.last_scan_time else None
            ),
            "scan_count": self.scan_count,
            "total_signals_generated": self.total_signals_generated,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "scan_interval": self.scan_interval,
            "top_n_signals": self.top_n_signals,
            "auto_trading_enabled": self.auto_trading_enabled,
            "notification_enabled": self.notification_enabled,
            "exchange": self.trading_engine.exchange_name or self.trading_engine.exchange,
        }

    # ------------------------------------------------------------------ #
    #  MANUAL SCAN
    # ------------------------------------------------------------------ #
    def force_scan(self) -> Dict[str, Any]:
        if not self.running:
            return {"success": False, "error": "Automated trader not running"}

        logger.info("Force scan initiated")
        try:
            asyncio.run(self._execute_trading_cycle())
            return {
                "success": True,
                "message": "Force scan completed",
                "scan_time": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # pragma: no cover
            logger.error("Force-scan error: %s", exc)
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    #  SETTINGS UPDATE (runtime)
    # ------------------------------------------------------------------ #
    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        try:
            if "scan_interval" in new_settings:
                self.scan_interval = int(new_settings["scan_interval"])
            if "top_n_signals" in new_settings:
                self.top_n_signals = int(new_settings["top_n_signals"])
            if "auto_trading_enabled" in new_settings:
                self.auto_trading_enabled = bool(new_settings["auto_trading_enabled"])
            if "notification_enabled" in new_settings:
                self.notification_enabled = bool(new_settings["notification_enabled"])
            logger.info("Automated trader settings updated")
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Error updating settings: %s", exc)
            return False