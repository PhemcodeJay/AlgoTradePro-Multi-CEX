import os
import json
import importlib
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from multi_trading_engine import TradingEngine
from db import db_manager
from logging_config import get_trading_logger

# Load environment variables
load_dotenv()
logger = get_trading_logger("settings")

SETTINGS_FILE = "settings.json"
ENV_FILE = ".env"

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def load_settings() -> dict:
    """Load settings from JSON file"""
    try:
        if not os.path.exists(SETTINGS_FILE):
            logger.warning(f"{SETTINGS_FILE} not found, creating defaults")
            return {}
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load settings: {e}", exc_info=True)
        return {}

def save_settings(settings: dict) -> bool:
    """Save settings to JSON file"""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save settings: {e}", exc_info=True)
        return False

def update_env_var(key: str, value: str):
    """Update .env file and reload environment"""
    try:
        env_vars = {}
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        env_vars[k] = v

        env_vars[key] = value
        with open(ENV_FILE, "w") as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")

        os.environ[key] = value
        load_dotenv(override=True)
        logger.info(f"Updated .env: {key}={value}")
    except Exception as e:
        logger.error(f"Failed to update .env: {e}", exc_info=True)

def get_client(exchange: str):
    """Dynamically import the right client"""
    try:
        if exchange.lower() == "binance":
            module = importlib.import_module("binance_client")
            return module.BinanceClient()
        elif exchange.lower() == "bybit":
            module = importlib.import_module("bybit_client")
            return module.BybitClient()
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    except Exception as e:
        logger.error(f"Failed to load client for {exchange}: {e}", exc_info=True)
        return None

# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

if "exchange" not in st.session_state:
    # Default from settings.json or .env
    settings = st.session_state.settings
    st.session_state.exchange = settings.get("EXCHANGE") or os.getenv("EXCHANGE", "binance")

if "client" not in st.session_state or "engine" not in st.session_state:
    st.session_state.client = get_client(st.session_state.exchange)
    st.session_state.engine = TradingEngine()

# -------------------------------------------------------------------
# Streamlit Settings Page
# -------------------------------------------------------------------

def render_settings_page():
    st.title("⚙️ Trading Settings")

    settings = st.session_state.settings
    exchange = st.session_state.exchange

    # Tabs for organization
    tabs = st.tabs(["General", "API", "Trading", "Risk", "Notifications", "Advanced"])

    # ----------------------------------------------------------------
    # General Tab
    # ----------------------------------------------------------------
    with tabs[0]:
        st.header("General Settings")

        scan_interval = st.number_input(
            "Scan Interval (seconds)",
            min_value=60,
            value=settings.get("SCAN_INTERVAL", 3600),
            step=60,
        )

        top_n_signals = st.number_input(
            "Top N Signals",
            min_value=1,
            value=settings.get("TOP_N_SIGNALS", 10),
            step=1,
        )

        symbols = st.text_area(
            "Symbols (comma separated)",
            value=",".join(settings.get("SYMBOLS", [])),
        )

        if st.button("Save General Settings"):
            settings["SCAN_INTERVAL"] = scan_interval
            settings["TOP_N_SIGNALS"] = top_n_signals
            settings["SYMBOLS"] = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            save_settings(settings)
            st.success("✅ General settings saved")

    # ----------------------------------------------------------------
    # API Tab
    # ----------------------------------------------------------------
    with tabs[1]:
        st.header("API & Exchange Settings")

        # Exchange selection
        new_exchange = st.selectbox(
            "Select Exchange",
            ["binance", "bybit"],
            index=["binance", "bybit"].index(exchange),
        )

        if new_exchange != exchange:
            # Update session + settings + env
            settings["EXCHANGE"] = new_exchange
            st.session_state.exchange = new_exchange
            save_settings(settings)
            update_env_var("EXCHANGE", new_exchange)

            # Reload client + engine
            st.session_state.client = get_client(new_exchange)
            st.session_state.engine = TradingEngine()

            st.success(f"✅ Switched to {new_exchange.capitalize()}")
            st.rerun()

        # API Keys
        api_key = st.text_input(
            f"{exchange.capitalize()} API Key",
            value=os.getenv(f"{exchange.upper()}_API_KEY", ""),
            type="password",
        )
        api_secret = st.text_input(
            f"{exchange.capitalize()} API Secret",
            value=os.getenv(f"{exchange.upper()}_API_SECRET", ""),
            type="password",
        )

        if st.button("Save API Keys"):
            update_env_var(f"{exchange.upper()}_API_KEY", api_key)
            update_env_var(f"{exchange.upper()}_API_SECRET", api_secret)
            st.success(f"✅ {exchange.capitalize()} API keys updated")

        # ----------------------------------------------------------------
    # Trading Tab
    # ----------------------------------------------------------------
    with tabs[2]:
        st.header("Trading Settings")

        leverage = st.number_input(
            "Leverage",
            min_value=1,
            max_value=125,
            value=int(settings.get("LEVERAGE", 10)),
            step=1,
        )

        max_positions = st.number_input(
            "Max Positions",
            min_value=1,
            max_value=50,
            value=int(settings.get("MAX_POSITIONS", 5)),
            step=1,
        )

        auto_trading = st.checkbox(
            "Enable Auto Trading",
            value=settings.get("AUTO_TRADING_ENABLED", True),
        )

        use_websocket = st.checkbox(
            "Use WebSocket for Realtime Data",
            value=settings.get("USE_WEBSOCKET", True),
        )

        if st.button("Save Trading Settings"):
            settings["LEVERAGE"] = leverage
            settings["MAX_POSITIONS"] = max_positions
            settings["AUTO_TRADING_ENABLED"] = auto_trading
            settings["USE_WEBSOCKET"] = use_websocket
            save_settings(settings)
            st.success("✅ Trading settings saved")

    # ----------------------------------------------------------------
    # Risk Tab
    # ----------------------------------------------------------------
    with tabs[3]:
        st.header("Risk Management")

        risk_pct = st.number_input(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=100.0,
            value=0.1,
            step=0.1,
        )

        tp_percent = st.number_input(
            "Take Profit (%)",
            min_value=0.1,
            max_value=100.0,
            value=settings.get("TP_PERCENT", 2.0),
            step=0.1,
        )

        sl_percent = st.number_input(
            "Stop Loss (%)",
            min_value=0.1,
            max_value=100.0,
            value=settings.get("SL_PERCENT", 1.5),
            step=0.1,
        )

        max_loss_pct = st.number_input(
            "Max Loss (%)",
            min_value=-100.0,
            max_value=-1.0,
            value=settings.get("MAX_LOSS_PCT", -15.0),
            step=0.1,
        )

        max_drawdown_pct = st.number_input(
            "Max Drawdown (%)",
            min_value=-100.0,
            max_value=-1.0,
            value=settings.get("MAX_DRAWDOWN_PCT", -20.0),
            step=0.1,
        )

        if st.button("Save Risk Settings"):
            settings["RISK_PCT"] = risk_pct
            settings["TP_PERCENT"] = tp_percent
            settings["SL_PERCENT"] = sl_percent
            settings["MAX_LOSS_PCT"] = max_loss_pct
            settings["MAX_DRAWDOWN_PCT"] = max_drawdown_pct
            save_settings(settings)
            st.success("✅ Risk settings saved")

    # ----------------------------------------------------------------
    # Notifications Tab
    # ----------------------------------------------------------------
    with tabs[4]:
        st.header("Notification Settings")

        notifications = st.checkbox(
            "Enable Notifications",
            value=settings.get("NOTIFICATION_ENABLED", True),
        )

        discord_webhook = st.text_input(
            "Discord Webhook URL",
            value=os.getenv("DISCORD_WEBHOOK_URL", ""),
        )

        telegram_bot = st.text_input(
            "Telegram Bot Token",
            value=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        )

        telegram_chat = st.text_input(
            "Telegram Chat ID",
            value=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

        whatsapp_to = st.text_input(
            "WhatsApp Number",
            value=os.getenv("WHATSAPP_TO", ""),
        )

        if st.button("Save Notification Settings"):
            settings["NOTIFICATION_ENABLED"] = notifications
            save_settings(settings)
            if discord_webhook:
                update_env_var("DISCORD_WEBHOOK_URL", discord_webhook)
            if telegram_bot:
                update_env_var("TELEGRAM_BOT_TOKEN", telegram_bot)
            if telegram_chat:
                update_env_var("TELEGRAM_CHAT_ID", telegram_chat)
            if whatsapp_to:
                update_env_var("WHATSAPP_TO", whatsapp_to)
            st.success("✅ Notification settings saved")
    # ----------------------------------------------------------------
    # Advanced Tab
    # ----------------------------------------------------------------
    with tabs[5]:
        st.header("Advanced Settings")

        virtual_balance = st.number_input(
            "Virtual Balance (for backtests / paper trading)",
            min_value=0.0,
            value=settings.get("VIRTUAL_BALANCE", 100.0),
            step=10.0,
        )

        entry_buffer = st.number_input(
            "Entry Buffer (%)",
            min_value=0.0,
            max_value=5.0,
            value=settings.get("ENTRY_BUFFER_PCT", 0.002),
            step=0.001,
        )

        min_atr_pct = st.number_input(
            "Min ATR (%)",
            min_value=0.0,
            max_value=100.0,
            value=settings.get("MIN_ATR_PCT", 0.5),
            step=0.1,
        )

        max_spread_pct = st.number_input(
            "Max Spread (%)",
            min_value=0.0,
            max_value=10.0,
            value=settings.get("MAX_SPREAD_PCT", 0.1),
            step=0.1,
        )

        ml_enabled = st.checkbox(
            "Enable Machine Learning",
            value=settings.get(settings["EXCHANGE"], {}).get("ML_ENABLED", True),
        )

        ml_retrain_threshold = st.number_input(
            "ML Retrain Threshold",
            min_value=10,
            max_value=10000,
            value=settings.get(settings["EXCHANGE"], {}).get("ML_RETRAIN_THRESHOLD", 100),
            step=10,
        )

        if st.button("Save Advanced Settings"):
            settings["VIRTUAL_BALANCE"] = virtual_balance
            settings["ENTRY_BUFFER_PCT"] = entry_buffer
            settings["MIN_ATR_PCT"] = min_atr_pct
            settings["MAX_SPREAD_PCT"] = max_spread_pct
            settings[settings["EXCHANGE"]]["ML_ENABLED"] = ml_enabled
            settings[settings["EXCHANGE"]]["ML_RETRAIN_THRESHOLD"] = ml_retrain_threshold
            save_settings(settings)
            st.success("✅ Advanced settings saved")

    # ----------------------------------------------------------------
    # Footer
    # ----------------------------------------------------------------
    st.markdown("---")
    st.caption("⚡ AlgoTrader Pro - Multi Exchange (Binance & Bybit) v2.0 | Settings Dashboard")

# ----------------------------------------------------------------
# Main Entrypoint
# ----------------------------------------------------------------
if __name__ == "__main__":
    render_settings_page()
