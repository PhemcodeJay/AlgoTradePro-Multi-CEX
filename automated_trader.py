# automated_trader.py (modified to pass exchange to signal_generator functions)
import asyncio
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from signal_generator import generate_signals, get_symbols, get_signal_summary
from ml import MLFilter
from notifications import send_all_notifications
from logging_config import get_trading_logger
from exceptions import TradingException, create_error_context

logger = get_trading_logger('automated')

class AutomatedTrader:
    def __init__(self, trading_engine):
        self.trading_engine = trading_engine
        self.running = False
        self.thread = None
        self.last_scan_time = None
        self.scan_count = 0
        self.total_signals_generated = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.scan_interval = 3600
        self.top_n_signals = 10
        self.auto_trading_enabled = False
        self.notification_enabled = True
        self._load_settings()
        logger.info(f"AutomatedTrader initialized for {self.trading_engine.exchange_name}")

    def _load_settings(self):
        """Load settings from trading engine"""
        try:
            self.scan_interval, self.top_n_signals = self.trading_engine.get_settings()
            settings = self.trading_engine.settings
            self.auto_trading_enabled = settings.get("AUTO_TRADING_ENABLED", False)
            self.notification_enabled = settings.get("NOTIFICATION_ENABLED", True)
            logger.info(f"Settings loaded: scan_interval={self.scan_interval}s, top_n={self.top_n_signals}")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")

    def start(self) -> bool:
        """Start the automated trading loop"""
        try:
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
        except Exception as e:
            logger.error(f"Failed to start automated trader: {e}")
            return False

    def stop(self) -> bool:
        """Stop the automated trading loop"""
        try:
            if not self.running:
                logger.warning("Automated trader is not running")
                return False
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10)
            logger.info("Automated trader stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop automated trader: {e}")
            return False
    
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Automated trading loop started")
        while self.running:
            try:
                if not self.trading_engine.is_trading_enabled():
                    logger.warning("Trading engine disabled, stopping trading loop")
                    self.stop()
                    return
                asyncio.run(self._execute_trading_cycle())
                time.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)

    async def _execute_trading_cycle(self):
        """Execute one cycle of trading operations using real market data"""
        try:
            self.scan_count += 1
            self.last_scan_time = datetime.now(timezone.utc)
            symbols = get_symbols(self.trading_engine.exchange_name, limit=50)
            if not symbols:
                logger.warning("No symbols retrieved for trading")
                return
            signals = generate_signals(
                self.trading_engine.exchange_name,
                symbols,
                interval="60",
                top_n=self.top_n_signals,
                user_id=self.trading_engine.user_id
            )
            self.total_signals_generated += len(signals)
            if not signals:
                logger.info("No valid signals generated")
                return
            if self.notification_enabled:
                for signal in signals:
                    send_all_notifications(signal)
            if self.auto_trading_enabled:
                for signal in signals:
                    try:
                        symbol = signal.get('symbol', '').replace('/', '')
                        signal['symbol'] = symbol
                        success = await self.trading_engine.execute_trade(signal)
                        if success:
                            self.successful_trades += 1
                            logger.info(f"Trade executed for {symbol}")
                        else:
                            self.failed_trades += 1
                            logger.warning(f"Trade execution failed for {symbol}")
                    except Exception as e:
                        self.failed_trades += 1
                        logger.error(f"Error executing trade for {signal.get('symbol')}: {e}")
            self._update_ml_feedback()
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")

    def _update_ml_feedback(self):
        """Update ML model with trade outcomes"""
        try:
            recent_trades = self.trading_engine.db.get_trades(
                hours=24,
                exchange=self.trading_engine.exchange_name,
                user_id=self.trading_engine.user_id
            )
            if not recent_trades:
                return
            ml_filter = MLFilter()
            for trade in recent_trades:
                try:
                    signal = {
                        'symbol': trade.symbol,
                        'indicators': trade.indicators or {},
                        'pnl': trade.pnl or 0
                    }
                    outcome = (trade.pnl or 0) > 0
                    ml_filter.update_model_with_feedback(signal, outcome)
                except Exception as e:
                    logger.error(f"Error updating ML feedback for trade {trade.id}: {e}")
            logger.info(f"Updated ML model with {len(recent_trades)} trade outcomes")
        except Exception as e:
            logger.error(f"Error updating ML model: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of automated trader"""
        return {
            'running': self.running,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'scan_count': self.scan_count,
            'total_signals_generated': self.total_signals_generated,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'scan_interval': self.scan_interval,
            'top_n_signals': self.top_n_signals,
            'auto_trading_enabled': self.auto_trading_enabled,
            'notification_enabled': self.notification_enabled,
            'exchange': self.trading_engine.exchange_name
        }

    def force_scan(self) -> Dict[str, Any]:
        """Force an immediate scan (for manual testing)"""
        try:
            if not self.running:
                return {'success': False, 'error': 'Automated trader not running'}
            logger.info("Force scan initiated")
            asyncio.run(self._execute_trading_cycle())
            return {
                'success': True,
                'message': 'Force scan completed',
                'scan_time': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error in force scan: {e}")
            return {'success': False, 'error': str(e)}

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update automated trader settings"""
        try:
            if 'scan_interval' in new_settings:
                self.scan_interval = int(new_settings['scan_interval'])
            if 'top_n_signals' in new_settings:
                self.top_n_signals = int(new_settings['top_n_signals'])
            if 'auto_trading_enabled' in new_settings:
                self.auto_trading_enabled = bool(new_settings['auto_trading_enabled'])
            if 'notification_enabled' in new_settings:
                self.notification_enabled = bool(new_settings['notification_enabled'])
            logger.info("Automated trader settings updated")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False