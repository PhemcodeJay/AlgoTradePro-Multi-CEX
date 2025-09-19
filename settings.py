import json
import os
from typing import Dict, Any
from logging_config import get_logger

logger = get_logger(__name__)

SETTINGS_FILE = "settings.json"

def load_settings() -> Dict[str, Any]:
    default_settings = {
        "SCAN_INTERVAL": int(os.getenv("DEFAULT_SCAN_INTERVAL", 3600)),
        "TOP_N_SIGNALS": int(os.getenv("DEFAULT_TOP_N_SIGNALS", 5)),
        "MAX_LOSS_PCT": -15.0,
        "TP_PERCENT": 0.15,
        "SL_PERCENT": 0.05,
        "MAX_DRAWDOWN_PCT": -15.0,
        "LEVERAGE": float(os.getenv("LEVERAGE", 10)),
        "RISK_PCT": float(os.getenv("RISK_PCT", 0.01)),
        "VIRTUAL_BALANCE": 100.0,
        "ENTRY_BUFFER_PCT": float(os.getenv("ENTRY_BUFFER_PCT", 0.002)),
        "SYMBOLS": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT"],
        "USE_WEBSOCKET": True,
        "MAX_POSITIONS": 5,
        "MIN_SIGNAL_SCORE": 60,
        "EXCHANGE": os.getenv("EXCHANGE", "binance")  # 👈 add exchange switch
    }

    try:
        if not os.path.exists(SETTINGS_FILE):
            logger.warning("settings.json not found, creating with defaults")
            with open(SETTINGS_FILE, "w") as f:
                json.dump(default_settings, f, indent=2)
            return default_settings

        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)

        # Merge defaults for missing keys
        for key, value in default_settings.items():
            if key not in settings:
                logger.warning(f"Missing {key} in {SETTINGS_FILE}, using default {value}")
                settings[key] = value
            else:
                try:
                    if isinstance(value, (int, float)):
                        settings[key] = float(settings[key])
                        if key in ["LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT"] and settings[key] <= 0:
                            logger.warning(f"Invalid {key} {settings[key]}, using default {value}")
                            settings[key] = value
                        if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and settings[key] > 0:
                            logger.warning(f"Invalid {key} {settings[key]}, using default {value}")
                            settings[key] = value

                    if key == "TOP_N_SIGNALS":
                        settings[key] = int(settings[key])
                        if settings[key] <= 0:
                            logger.warning(f"Invalid TOP_N_SIGNALS {settings[key]}, using default {value}")
                            settings[key] = value
                except (ValueError, TypeError):
                    logger.warning(f"Invalid {key} in {SETTINGS_FILE}, using default {value}")
                    settings[key] = value

        logger.info("Settings loaded successfully")
        return settings

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding {SETTINGS_FILE}: {e}, using defaults")
        return default_settings
    except Exception as e:
        logger.error(f"Error loading {SETTINGS_FILE}: {e}, using defaults")
        return default_settings


def save_settings(settings: Dict[str, Any]) -> bool:
    try:
        for key, value in settings.items():
            if key in ["LEVERAGE", "RISK_PCT", "VIRTUAL_BALANCE", "ENTRY_BUFFER_PCT"] and float(value) <= 0:
                logger.error(f"Invalid {key}: {value} must be positive")
                return False
            if key in ["MAX_LOSS_PCT", "MAX_DRAWDOWN_PCT"] and float(value) > 0:
                logger.error(f"Invalid {key}: {value} must be negative")
                return False

        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        logger.info("Settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False


def validate_env(exchange: str | None = None) -> bool:
    """
    Validate required environment variables depending on exchange.
    Defaults to settings.json EXCHANGE if not provided.
    """
    settings = load_settings()
    # ensure exchange is always str
    exchange_val = exchange if exchange is not None else settings.get("EXCHANGE", "binance")
    exchange = str(exchange_val).lower()

    required_vars = []
    if exchange == "binance":
        required_vars = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
    elif exchange == "bybit":
        required_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]

    optional_vars = [
        "DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DATABASE_URL"
    ]

    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        logger.error(f"Missing required env vars for {exchange}: {', '.join(missing_required)}")
        return False

    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    if missing_optional:
        logger.warning(f"Missing optional env vars: {', '.join(missing_optional)}")

    return True
