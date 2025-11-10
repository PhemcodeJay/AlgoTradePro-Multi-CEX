import streamlit as st
import bcrypt
from datetime import datetime, timezone
from db import db_manager, User
from settings import load_settings, save_settings, validate_env
from logging_config import get_trading_logger
from multi_trading_engine import TradingEngine

# Set page configuration
st.set_page_config(
    page_title="Settings - AlgoTraderPro V2.0",
    page_icon="⚙️",
    layout="wide"
)

logger = get_trading_logger(__name__)

# --- Authentication and Engine Initialization Checks ---
# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Redirect if not authenticated
if not st.session_state.get('authenticated', False):
    st.warning("⚠️ Please login from the main page to access settings.")
    st.stop()

# Ensure the trading engine is initialized
if "trading_engine" not in st.session_state or not st.session_state.trading_engine:
    st.error("Trading engine not initialized – go to the dashboard first.")
    if st.button("Go to Dashboard"):
        st.switch_page("app.py")
    st.stop()

# --- Get essential session state variables ---
trading_engine: TradingEngine = st.session_state.trading_engine
user_id = st.session_state.user["id"]
current_exchange = st.session_state.current_exchange
account_type = st.session_state.account_type

# --- Header ---
st.markdown("""
<div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0;'>⚙️ Settings</h1>
    <p style='color: white; margin: 0.5rem 0 0 0;'>Configure your trading bot</p>
</div>
""", unsafe_allow_html=True)

# --- Load current settings ---
try:
    current_settings = load_settings(user_id=user_id) or {}
except Exception as e:
    logger.error(f"Settings load error: {e}")
    st.error("Failed to load settings")
    current_settings = {}

# --- Get API keys from database ---
with db_manager.get_session() as s:
    db_user = s.query(User).get(user_id)
    binance_key = db_user.binance_api_key or ""
    binance_secret = db_user.binance_api_secret or ""
    bybit_key = db_user.bybit_api_key or ""
    bybit_secret = db_user.bybit_api_secret or ""

# --- Tabs for different setting categories ---
tab1, tab2, tab3, tab4 = st.tabs(["API Keys", "Trading", "Risk Management", "Notifications"])

# --- Tab 1: API Keys ---
with tab1:
    st.markdown("### 🔑 API Credentials")
    st.info("Enter your exchange API keys to enable real trading")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Binance**")
        binance_api_key = st.text_input(
            "API Key",
            value=binance_key,
            type="password",
            key="binance_key"
        )
        binance_api_secret = st.text_input(
            "API Secret",
            value=binance_secret,
            type="password",
            key="binance_secret"
        )

    with col2:
        st.markdown("**Bybit**")
        bybit_api_key = st.text_input(
            "API Key",
            value=bybit_key,
            type="password",
            key="bybit_key"
        )
        bybit_api_secret = st.text_input(
            "API Secret",
            value=bybit_secret,
            type="password",
            key="bybit_secret"
        )

    if st.button("💾 Save API Keys", type="primary"):
        try:
            with db_manager.get_session() as session:
                user = session.query(User).get(user_id)
                if user:
                    # Save keys, setting to None if input is empty
                    user.binance_api_key = binance_api_key if binance_api_key else None
                    user.binance_api_secret = binance_api_secret if binance_api_secret else None
                    user.bybit_api_key = bybit_api_key if bybit_api_key else None
                    user.bybit_api_secret = bybit_api_secret if bybit_api_secret else None
                    session.commit()
                    st.success("✅ API keys saved successfully!")
                    # Rerun to reflect changes immediately
                    st.rerun()
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            st.error(f"Failed to save API keys: {e}")

# --- Tab 2: Trading Settings ---
with tab2:
    st.markdown("### 📊 Trading Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Scan Interval
        scan_interval = st.number_input(
            "Scan Interval (seconds)",
            min_value=60,
            max_value=86400,
            value=int(current_settings.get("SCAN_INTERVAL", 3600)),
            step=60
        )

        # Top N Signals
        top_n_signals = st.number_input(
            "Top N Signals",
            min_value=1,
            max_value=50,
            value=int(current_settings.get("TOP_N_SIGNALS", 10))
        )

        # Leverage
        leverage = st.slider(
            "Leverage",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("LEVERAGE", 10))
        )

    with col2:
        # Auto Trading Enabled
        auto_trading = st.checkbox(
            "Auto Trading Enabled",
            value=bool(current_settings.get("AUTO_TRADING_ENABLED", False))
        )

        # ML Filtering Enabled
        ml_enabled = st.checkbox(
            "ML Filtering Enabled",
            value=bool(current_settings.get("ML_ENABLED", True))
        )

        # Minimum Signal Score
        min_signal_score = st.slider(
            "Minimum Signal Score",
            min_value=0.0,
            max_value=100.0,
            value=float(current_settings.get("MIN_SIGNAL_SCORE", 40.0))
        )

    # Save Trading Settings button
    if st.button("💾 Save Trading Settings", type="primary"):
        try:
            new_settings = {
                "SCAN_INTERVAL": scan_interval,
                "TOP_N_SIGNALS": top_n_signals,
                "LEVERAGE": leverage,
                "AUTO_TRADING_ENABLED": auto_trading,
                "ML_ENABLED": ml_enabled,
                "MIN_SIGNAL_SCORE": min_signal_score
            }
            save_settings(new_settings, user_id)
            st.success("✅ Trading settings saved!")
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            st.error(f"Failed to save settings: {e}")

# --- Tab 3: Risk Management ---
with tab3:
    st.markdown("### 🛡️ Risk Management")

    col1, col2 = st.columns(2)

    with col1:
        # Risk Percentage per Trade
        risk_pct = st.slider(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(current_settings.get("RISK_PCT", 2.0)), # Assuming RISK_PCT is stored as decimal, adjust if it's percentage
            step=0.1
        )

        # Max Open Positions
        max_positions = st.number_input(
            "Max Open Positions",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("MAX_OPEN_POSITIONS", 10))
        )

        # Max Daily Loss
        max_daily_loss = st.number_input(
            "Max Daily Loss (USDT)",
            min_value=10.0,
            max_value=10000.0,
            value=float(current_settings.get("MAX_DAILY_LOSS", 1000.0)),
            step=10.0
        )

    with col2:
        # Take Profit Percentage
        tp_percent = st.slider(
            "Take Profit (%)",
            min_value=0.5,
            max_value=10.0,
            value=float(current_settings.get("TP_PERCENT", 2.0)),
            step=0.1
        )

        # Stop Loss Percentage
        sl_percent = st.slider(
            "Stop Loss (%)",
            min_value=0.5,
            max_value=10.0,
            value=float(current_settings.get("SL_PERCENT", 1.5)),
            step=0.1
        )

        # Max Position Size
        max_position_size = st.number_input(
            "Max Position Size (USDT)",
            min_value=100.0,
            max_value=100000.0,
            value=float(current_settings.get("MAX_POSITION_SIZE", 10000.0)),
            step=100.0
        )

    # Save Risk Settings button
    if st.button("💾 Save Risk Settings", type="primary"):
        try:
            new_settings = {
                "RISK_PCT": risk_pct / 100, # Convert percentage to decimal for saving if needed
                "MAX_OPEN_POSITIONS": max_positions,
                "MAX_DAILY_LOSS": max_daily_loss,
                "TP_PERCENT": tp_percent,
                "SL_PERCENT": sl_percent,
                "MAX_POSITION_SIZE": max_position_size
            }
            save_settings(new_settings, user_id)
            st.success("✅ Risk settings saved!")
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            st.error(f"Failed to save settings: {e}")

# --- Tab 4: Notification Settings ---
with tab4:
    st.markdown("### 🔔 Notification Settings")

    # Enable Notifications checkbox
    notification_enabled = st.checkbox(
        "Enable Notifications",
        value=bool(current_settings.get("NOTIFICATION_ENABLED", True))
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Discord Signal Limit
        discord_limit = st.number_input(
            "Discord Signal Limit",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("DISCORD_SIGNAL_LIMIT", 5))
        )

    with col2:
        # Telegram Signal Limit
        telegram_limit = st.number_input(
            "Telegram Signal Limit",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("TELEGRAM_SIGNAL_LIMIT", 5))
        )

    with col3:
        # WhatsApp Signal Limit
        whatsapp_limit = st.number_input(
            "WhatsApp Signal Limit",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("WHATSAPP_SIGNAL_LIMIT", 3))
        )

    # Save Notification Settings button
    if st.button("💾 Save Notification Settings", type="primary"):
        try:
            new_settings = {
                "NOTIFICATION_ENABLED": notification_enabled,
                "DISCORD_SIGNAL_LIMIT": discord_limit,
                "TELEGRAM_SIGNAL_LIMIT": telegram_limit,
                "WHATSAPP_SIGNAL_LIMIT": whatsapp_limit
            }
            save_settings(new_settings, user_id)
            st.success("✅ Notification settings saved!")
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            st.error(f"Failed to save settings: {e}")

# --- Footer with last updated timestamp ---
st.markdown(f"""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
</div>
""", unsafe_allow_html=True)