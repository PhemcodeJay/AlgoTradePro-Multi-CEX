import json
import os
from typing import Dict, Any, Optional, Tuple
from logging_config import get_logger
from db import db_manager, User, Settings

logger = get_logger(__name__)

SETTINGS_FILE = "settings.json"

def load_settings(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Load settings from JSON (global) or database (user-specific)."""
    default_settings = {
        "SCAN_INTERVAL": float(os.getenv("DEFAULT_SCAN_INTERVAL", 3600.0)),
        "TOP_N_SIGNALS": int(os.getenv("DEFAULT_TOP_N_SIGNALS", 10)),
        "MAX_LOSS_PCT": -15.0,
        "TP_PERCENT": 2.0,
        "SL_PERCENT": 1.5,
        "MAX_DRAWDOWN_PCT": -20.0,
        "LEVERAGE": float(os.getenv("LEVERAGE", 10.0)),
        "RISK_PCT": float(os.getenv("RISK_PCT", 0.02)),
        "VIRTUAL_BALANCE": 100.0,
        "ENTRY_BUFFER_PCT": float(os.getenv("ENTRY_BUFFER_PCT", 0.002)),
        "SYMBOLS": [
            "BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT",
            "BNBUSDT", "AVAXUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"
        ],
        "USE_WEBSOCKET": True,
        "MAX_POSITIONS": 5,
        "MIN_SIGNAL_SCORE": 40.0,
        "EXCHANGE": os.getenv("EXCHANGE", "binance"),
        "AUTO_TRADING_ENABLED": False,
        "NOTIFICATION_ENABLED": True,
        "RSI_OVERSOLD": 30.0,
        "RSI_OVERBOUGHT": 70.0,
        "MIN_VOLUME": 1000000.0,
        "MIN_ATR_PCT": 0.5,
        "MAX_SPREAD_PCT": 0.1,
        "ML_ENABLED": True,
        "ML_RETRAIN_THRESHOLD": 100.0,
        "MAX_POSITION_SIZE": 10000.0,
        "MAX_OPEN_POSITIONS": 10,
        "MAX_DAILY_LOSS": 1000.0,
        "MAX_RISK_PER_TRADE": 0.05,
        "DISCORD_SIGNAL_LIMIT": 5,
        "TELEGRAM_SIGNAL_LIMIT": 5,
        "WHATSAPP_SIGNAL_LIMIT": 3,
        "binance": {
            "AUTO_TRADING_ENABLED": False,
            "RSI_OVERSOLD": 30.0,
            "RSI_OVERBOUGHT": 70.0,
            "MIN_VOLUME": 1000000.0,
            "NOTIFICATION_ENABLED": True,
            "ML_ENABLED": True,
            "ML_RETRAIN_THRESHOLD": 100.0
        },
        "bybit": {
            "AUTO_TRADING_ENABLED": False,
            "RSI_OVERSOLD": 25.0,
            "RSI_OVERBOUGHT": 75.0,
            "MIN_VOLUME": 500000.0,
            "NOTIFICATION_ENABLED": False,
            "ML_ENABLED": True,
            "ML_RETRAIN_THRESHOLD": 200.0
        }
    }

    # GLOBAL (settings.json)
    if user_id is None:
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE) as f:
                    file_settings = json.load(f)
                return {**default_settings, **file_settings}
            except Exception as e:
                logger.error(f"Failed to load global settings: {e}")
        return default_settings

    # USER-SPECIFIC (DB)
    settings = default_settings.copy()
    with db_manager.get_session() as session:
        try:
            # Load user row
            user = session.query(User).get(user_id)
            if user:
                settings["EXCHANGE"] = getattr(user, 'default_exchange', None) or "bybit"
                settings["TRADING_MODE"] = getattr(user, 'trading_mode', None) or "virtual"

            # Load Settings table (key-value overrides)
            rows = session.query(Settings).filter_by(user_id=user_id).all()
            for row in rows:
                key = row.key
                raw = row.value
                default = default_settings.get(key)

                if default is None:
                    settings[key] = raw
                    continue

                if isinstance(default, bool):
                    if isinstance(raw, bool):
                        settings[key] = raw
                    elif isinstance(raw, str):
                        settings[key] = raw.strip().lower() in ("true", "1", "yes", "on")
                    else:
                        settings[key] = bool(raw)
                elif isinstance(default, (int, float)):
                    try:
                        settings[key] = type(default)(raw)
                    except (ValueError, TypeError):
                        settings[key] = default  # fallback
                elif isinstance(default, list):
                    try:
                        settings[key] = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        settings[key] = default
                elif isinstance(default, dict):
                    try:
                        settings[key] = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        settings[key] = default
                else:
                    settings[key] = raw

        except Exception as e:
            logger.error(f"Error loading user settings: {e}")

    return settings


def save_settings(settings: Dict[str, Any], user_id: Optional[int] = None) -> bool:
    """Save settings + API keys per user."""
    if user_id is None:
        # Global save
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(settings, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save global settings: {e}")
            return False

    # USER-SPECIFIC
    with db_manager.get_session() as session:
        try:
            user = session.query(User).get(user_id)
            if not user:
                logger.error(f"User {user_id} not found")
                return False

            # === SAVE API KEYS TO User TABLE ===
            if "BINANCE_API_KEY" in settings:
                user.binance_api_key = settings.pop("BINANCE_API_KEY") or None
            if "BINANCE_API_SECRET" in settings:
                user.binance_api_secret = settings.pop("BINANCE_API_SECRET") or None
            if "BYBIT_API_KEY" in settings:
                user.bybit_api_key = settings.pop("BYBIT_API_KEY") or None
            if "BYBIT_API_SECRET" in settings:
                user.bybit_api_secret = settings.pop("BYBIT_API_SECRET") or None

            # Optional: save exchange & mode
            if "EXCHANGE" in settings:
                user.default_exchange = settings["EXCHANGE"]
            if "TRADING_MODE" in settings:
                user.trading_mode = settings["TRADING_MODE"]

            # === SAVE OTHER SETTINGS TO Settings TABLE ===
            for key, value in settings.items():
                if isinstance(value, (list, dict, bool)):
                    value = json.dumps(value)

                row = session.query(Settings).filter_by(user_id=user_id, key=key).first()
                if row:
                    row.value = value
                else:
                    session.add(Settings(user_id=user_id, key=key, value=value))

            session.commit()
            logger.info(f"Settings + API keys saved for user {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save user settings: {e}")
            return False


def validate_env(exchange: Optional[str] = None, user_id: Optional[int] = None) -> Tuple[bool, Dict]:
    """Check API keys in DB only. Returns (has_keys, {api_key, api_secret})"""
    if not user_id:
        logger.error("validate_env requires user_id")
        return False, {}

    with db_manager.get_session() as s:
        user = s.query(User).get(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return False, {}

        exchange_name = (exchange or getattr(user, 'default_exchange', None) or "bybit").lower()

        if exchange_name == "binance":
            key = user.binance_api_key
            secret = user.binance_api_secret
        elif exchange_name == "bybit":
            key = user.bybit_api_key
            secret = user.bybit_api_secret
        else:
            logger.error(f"Unknown exchange: {exchange_name}")
            return False, {}

        has_keys = bool(key and secret)
        return has_keys, {"api_key": key or "", "api_secret": secret or ""}