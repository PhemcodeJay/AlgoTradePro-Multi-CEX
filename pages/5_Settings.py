import streamlit as st
import json
import os
from typing import Dict, Any
from datetime import datetime, timezone
from db import db_manager, User, WalletBalance
from settings import load_settings, save_settings, validate_env
from logging_config import get_trading_logger
import bcrypt

# Page configuration
st.set_page_config(
    page_title="Settings - AlgoTraderPro V2.0",
    page_icon="⚙️",
    layout="wide"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Authentication functions
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user against the database with hashed password."""
    try:
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                st.session_state.user = {'id': user.id, 'username': user.username}
                return True
            return False
    except Exception as e:
        logger.error(f"Authentication error for {username}: {e}")
        return False

def register_user(username: str, password: str) -> bool:
    """Register a new user with hashed password and initialize wallet balances."""
    try:
        with db_manager.get_session() as session:
            if session.query(User).filter_by(username=username).first():
                logger.warning(f"Registration failed: Username {username} already exists")
                return False
            
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            new_user = User(
                username=username,
                password_hash=hashed_password,
                binance_api_key=None,
                binance_api_secret=None,
                bybit_api_key=None,
                bybit_api_secret=None
            )
            session.add(new_user)
            session.commit()

            session.add(WalletBalance(
                user_id=new_user.id,
                account_type="virtual",
                available=100.0,
                used=0.0,
                total=100.0,
                currency="USDT",
                exchange=st.session_state.get('current_exchange', 'binance')
            ))
            session.add(WalletBalance(
                user_id=new_user.id,
                account_type="real",
                available=0.0,
                used=0.0,
                total=0.0,
                currency="USDT",
                exchange=st.session_state.get('current_exchange', 'binance')
            ))
            session.commit()
            logger.info(f"Registered new user: {username}, ID: {new_user.id}")
            return True
    except Exception as e:
        logger.error(f"Registration error for {username}: {e}")
        session.rollback()
        return False

# Login/Register UI
if not st.session_state.get('authenticated', False):
    st.markdown("### 🔐 Login or Register")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login")
            if submit_login:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_register = st.form_submit_button("Register")
            if submit_register:
                if new_password == confirm_password:
                    if register_user(new_username, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed. Username may already exist.")
                else:
                    st.error("Passwords do not match")

    st.stop()

# Initialize components
if 'trading_engine' not in st.session_state or st.session_state.trading_engine is None:
    st.error("Trading engine not initialized. Please return to the main page to initialize the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
current_exchange = st.session_state.get('current_exchange', 'binance')
account_type = st.session_state.get('account_type', 'virtual')
user_id = st.session_state.user.get('id') if st.session_state.user else None

if not user_id:
    st.error("User not authenticated. Please log in from the main page.")
    st.stop()

# Header with card
st.markdown("""
<div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0;'>⚙️ System Settings</h1>
    <p style='color: white; margin: 0.5rem 0 0 0;'>Configure your trading parameters and preferences</p>
</div>
""", unsafe_allow_html=True)

# Load user-specific settings
current_settings = load_settings(user_id=user_id) or {}

# Quick Info Cards
st.markdown("### 📋 Quick Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #667eea;'>
        <h4 style='margin: 0; color: #667eea;'>🏦 Exchange</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{current_exchange.title()}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #764ba2;'>
        <h4 style='margin: 0; color: #764ba2;'>🎯 Mode</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{account_type.title()}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    api_status = "✅ Connected" if validate_env(current_exchange, allow_virtual=False, user_id=user_id) else "🔒 Virtual Only"
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #10b981;'>
        <h4 style='margin: 0; color: #10b981;'>🔐 API Status</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{api_status}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Settings management tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trading",
    "🏦 Exchange",
    "⚠️ Risk Management",
    "🔔 Notifications",
    "🔧 System"
])

with tab1:
    st.markdown("### 📈 Trading Configuration")

    with st.expander("ℹ️ About Trading Settings", expanded=False):
        st.markdown("""
        **Scan Interval:** How often to scan markets for new opportunities

        **Top N Signals:** Maximum signals to generate per scan

        **Min Signal Score:** Only generate signals above this quality threshold

        **Auto Trading:** Automatically execute trades when enabled
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**General Settings**")

        scan_interval = st.number_input(
            "Scan Interval (seconds)",
            min_value=300,
            max_value=86400,
            value=int(current_settings.get("SCAN_INTERVAL", 3600)),
            step=300,
            help="How often to scan for new trading signals"
        )

        top_n_signals = st.number_input(
            "Top N Signals",
            min_value=1,
            max_value=50,
            value=current_settings.get("TOP_N_SIGNALS", 10),
            help="Maximum number of signals per scan"
        )

        min_signal_score = st.slider(
            "Minimum Signal Score",
            min_value=0,
            max_value=100,
            value=int(current_settings.get("MIN_SIGNAL_SCORE", 60)),
            step=5,
            help="Minimum score required for a signal to be considered"
        )

        auto_trading = st.checkbox(
            "Enable Auto Trading",
            value=current_settings.get("AUTO_TRADING_ENABLED", False),
            help="Automatically execute trades based on signals"
        )

        virtual_balance = st.number_input(
            "Virtual Trading Balance ($)",
            min_value=10.0,
            max_value=1000000.0,
            value=current_settings.get("VIRTUAL_BALANCE", 100.0),
            step=10.0,
            help="Starting balance for virtual trading"
        )

    with col2:
        st.markdown("**Technical Analysis**")

        rsi_oversold = st.number_input(
            "RSI Oversold Level",
            min_value=10,
            max_value=40,
            value=int(current_settings.get("RSI_OVERSOLD", 30))
        )

        rsi_overbought = st.number_input(
            "RSI Overbought Level",
            min_value=60,
            max_value=90,
            value=int(current_settings.get("RSI_OVERBOUGHT", 70))
        )

        min_volume = st.number_input(
            "Minimum Volume (USDT)",
            min_value=100000,
            max_value=10000000,
            value=int(current_settings.get("MIN_VOLUME", 1000000)),
            step=100000
        )

        min_atr_pct = st.number_input(
            "Minimum ATR (%)",
            min_value=0.1,
            max_value=5.0,
            value=current_settings.get("MIN_ATR_PCT", 0.5),
            step=0.1,
            format="%.1f"
        )

        max_spread_pct = st.number_input(
            "Maximum Spread (%)",
            min_value=0.01,
            max_value=1.0,
            value=current_settings.get("MAX_SPREAD_PCT", 0.1),
            step=0.01,
            format="%.2f"
        )

with tab2:
    st.markdown("### 🏦 Exchange Configuration")

    st.info("💡 **Tip:** Switch exchanges using the buttons at the top of the main page.")

    # Display API status for both exchanges
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='padding: 1rem; background: #fff8e1; border-radius: 8px; border: 2px solid #ffd700;'>
            <h3 style='margin: 0;'>🟡 Binance</h3>
        </div>
        """, unsafe_allow_html=True)

        binance_has_keys = validate_env("binance", allow_virtual=False, user_id=user_id)
        if binance_has_keys:
            st.success("✅ API Configured")
        else:
            st.warning("🔒 Virtual Mode Only")
            st.caption("Add Binance API keys below to enable real trading")

    with col2:
        st.markdown("""
        <div style='padding: 1rem; background: #fff3e0; border-radius: 8px; border: 2px solid #ff9800;'>
            <h3 style='margin: 0;'>🟠 Bybit</h3>
        </div>
        """, unsafe_allow_html=True)

        bybit_has_keys = validate_env("bybit", allow_virtual=False, user_id=user_id)
        if bybit_has_keys:
            st.success("✅ API Configured")
        else:
            st.warning("🔒 Virtual Mode Only")
            st.caption("Add Bybit API keys below to enable real trading")

    st.markdown("---")  # Separator

    # API Key Input Section
    with st.expander("🔑 API Credentials", expanded=True):
        st.markdown(f"### {current_exchange.upper()} API Keys")

        try:
            with db_manager.get_session() as session:
                user = session.query(User).filter_by(id=user_id).first()
                binance_api_key = user.binance_api_key or ""
                binance_api_secret = user.binance_api_secret or ""
                bybit_api_key = user.bybit_api_key or ""
                bybit_api_secret = user.bybit_api_secret or ""
        except Exception as e:
            logger.error(f"Error fetching user API keys: {e}")
            st.error(f"Error fetching API keys: {e}")
            binance_api_key = binance_api_secret = bybit_api_key = bybit_api_secret = ""

        if current_exchange == "binance":
            api_key = st.text_input(
                "Binance API Key",
                value=binance_api_key,
                type="password",
                key="binance_api_key_input"
            )
            api_secret = st.text_input(
                "Binance API Secret",
                value=binance_api_secret,
                type="password",
                key="binance_api_secret_input"
            )

            if st.button("💾 Save Binance API Keys", type="primary"):
                if api_key and api_secret:
                    try:
                        with db_manager.get_session() as session:
                            user = session.query(User).filter_by(id=user_id).first()
                            user.binance_api_key = api_key
                            user.binance_api_secret = api_secret
                            session.commit()
                            st.success("✅ Binance API keys saved successfully! Please restart the app for changes to take effect.")
                            st.info("Click the Stop button and then Run button to restart.")
                    except Exception as e:
                        logger.error(f"Error saving Binance API keys: {e}")
                        st.error(f"Error saving API keys: {e}")
                else:
                    st.error("Please provide both API Key and Secret")

        elif current_exchange == "bybit":
            api_key = st.text_input(
                "Bybit API Key",
                value=bybit_api_key,
                type="password",
                key="bybit_api_key_input"
            )
            api_secret = st.text_input(
                "Bybit API Secret",
                value=bybit_api_secret,
                type="password",
                key="bybit_api_secret_input"
            )

            if st.button("💾 Save Bybit API Keys", type="primary"):
                if api_key and api_secret:
                    try:
                        with db_manager.get_session() as session:
                            user = session.query(User).filter_by(id=user_id).first()
                            user.bybit_api_key = api_key
                            user.bybit_api_secret = api_secret
                            session.commit()
                            st.success("✅ Bybit API keys saved successfully! Please restart the app for changes to take effect.")
                            st.info("Click the Stop button and then Run button to restart.")
                    except Exception as e:
                        logger.error(f"Error saving Bybit API keys: {e}")
                        st.error(f"Error saving API keys: {e}")
                else:
                    st.error("Please provide both API Key and Secret")

    st.markdown("""
    ### API Configuration
    You can also use the Secrets tool (🔒) in the sidebar to add your API keys securely.
    """)

with tab3:
    st.markdown("### ⚠️ Risk Management")

    with st.expander("ℹ️ About Risk Management", expanded=False):
        st.markdown("""
        **Position Limits:** Control how much capital to allocate

        **Risk Parameters:** Define acceptable risk levels

        **Stop Loss/Take Profit:** Automatic exit points
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Position Limits**")

        max_positions = st.number_input(
            "Maximum Open Positions",
            min_value=1,
            max_value=50,
            value=int(current_settings.get("MAX_OPEN_POSITIONS", 5))
        )

        max_position_size = st.number_input(
            "Maximum Position Size ($)",
            min_value=10.0,
            max_value=100000.0,
            value=float(current_settings.get("MAX_POSITION_SIZE", 10000.0)),
            step=100.0
        )

        max_daily_loss = st.number_input(
            "Maximum Daily Loss ($)",
            min_value=10.0,
            max_value=10000.0,
            value=float(current_settings.get("MAX_DAILY_LOSS", 1000.0)),
            step=10.0
        )

    with col2:
        st.markdown("**Risk Parameters**")

        leverage = st.slider(
            "Default Leverage",
            min_value=1,
            max_value=20,
            value=int(current_settings.get("LEVERAGE", 10))
        )

        risk_pct = st.slider(
            "Risk % per Trade",
            min_value=0.1,
            max_value=10.0,
            value=current_settings.get("RISK_PCT", 0.02) * 100,
            step=0.1,
            format="%.1f%%"
        )

        tp_percent = st.number_input(
            "Default Take Profit (%)",
            min_value=0.5,
            max_value=20.0,
            value=current_settings.get("TP_PERCENT", 2.0),
            step=0.1,
            format="%.1f"
        )

        sl_percent = st.number_input(
            "Default Stop Loss (%)",
            min_value=0.5,
            max_value=20.0,
            value=current_settings.get("SL_PERCENT", 1.5),
            step=0.1,
            format="%.1f"
        )

with tab4:
    st.markdown("### 🔔 Notification Settings")

    # Notification Settings
    with st.expander("🔔 Configure Notifications", expanded=False):
        notification_enabled = st.checkbox(
            "Enable Notifications",
            value=current_settings.get("NOTIFICATION_ENABLED", True),
            help="Enable or disable all notifications"
        )

        discord_webhook = st.text_input(
            "Discord Webhook URL",
            value=current_settings.get("DISCORD_WEBHOOK_URL", ""),
            type="password",
            key="discord_webhook_input"
        )

        telegram_bot_token = st.text_input(
            "Telegram Bot Token",
            value=current_settings.get("TELEGRAM_BOT_TOKEN", ""),
            type="password",
            key="telegram_bot_token_input"
        )

        telegram_chat_id = st.text_input(
            "Telegram Chat ID",
            value=current_settings.get("TELEGRAM_CHAT_ID", ""),
            key="telegram_chat_id_input"
        )

        whatsapp_to = st.text_input(
            "WhatsApp To Number",
            value=current_settings.get("WHATSAPP_TO", ""),
            key="whatsapp_to_input"
        )

with tab5:
    st.markdown("### 🔧 System Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trading Symbols**")
        current_symbols = current_settings.get("SYMBOLS", [])
        symbols_text = st.text_area(
            "Symbols (one per line)",
            value="\n".join(current_symbols),
            height=200
        )

        ml_enabled = st.checkbox(
            "Enable ML Filtering",
            value=current_settings.get("ML_ENABLED", True)
        )

    with col2:
        st.markdown("**System Status**")

        status_data = {
            "Exchange": current_exchange.title(),
            "Trading Enabled": trading_engine.is_trading_enabled(),
            "Last Updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        }

        for key, value in status_data.items():
            st.markdown(f"**{key}:** {value}")

# Save settings
st.divider()
st.markdown("### 💾 Save Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        try:
            new_settings = {
                "SCAN_INTERVAL": scan_interval,
                "TOP_N_SIGNALS": top_n_signals,
                "MIN_SIGNAL_SCORE": min_signal_score,
                "AUTO_TRADING_ENABLED": auto_trading,
                "VIRTUAL_BALANCE": virtual_balance,
                "RSI_OVERSOLD": rsi_oversold,
                "RSI_OVERBOUGHT": rsi_overbought,
                "MIN_VOLUME": min_volume,
                "MIN_ATR_PCT": min_atr_pct,
                "MAX_SPREAD_PCT": max_spread_pct,
                "MAX_OPEN_POSITIONS": max_positions,
                "MAX_POSITION_SIZE": max_position_size,
                "MAX_DAILY_LOSS": max_daily_loss,
                "LEVERAGE": leverage,
                "RISK_PCT": risk_pct / 100,
                "TP_PERCENT": tp_percent,
                "SL_PERCENT": sl_percent,
                "NOTIFICATION_ENABLED": notification_enabled,
                "ML_ENABLED": ml_enabled,
                "SYMBOLS": [s.strip().upper() for s in symbols_text.split('\n') if s.strip()],
                "EXCHANGE": current_exchange,
                "DISCORD_WEBHOOK_URL": discord_webhook,
                "TELEGRAM_BOT_TOKEN": telegram_bot_token,
                "TELEGRAM_CHAT_ID": telegram_chat_id,
                "WHATSAPP_TO": whatsapp_to
            }

            success = save_settings(new_settings, user_id=user_id)

            if success:
                # Update virtual balance in WalletBalance if changed
                if virtual_balance != current_settings.get("VIRTUAL_BALANCE", 100.0):
                    try:
                        with db_manager.get_session() as session:
                            wallet = session.query(WalletBalance).filter_by(
                                user_id=user_id,
                                account_type="virtual",
                                exchange=current_exchange
                            ).first()
                            if wallet:
                                wallet.available = virtual_balance
                                wallet.total = virtual_balance
                                session.commit()
                    except Exception as e:
                        logger.error(f"Error updating virtual balance: {e}")
                        st.error(f"Error updating virtual balance: {e}")

                st.success("✅ Settings saved successfully!")
                trading_engine.reload_settings()  # Assuming the engine has a reload method
                st.balloons()
            else:
                st.error("❌ Failed to save settings")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            st.error(f"Error: {e}")

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        default_settings = {
            "SCAN_INTERVAL": 3600,
            "TOP_N_SIGNALS": 5,
            "MIN_SIGNAL_SCORE": 60,
            "AUTO_TRADING_ENABLED": False,
            "VIRTUAL_BALANCE": 100.0,
            "RSI_OVERSOLD": 30,
            "RSI_OVERBOUGHT": 70,
            "MIN_VOLUME": 1000000,
            "MIN_ATR_PCT": 0.5,
            "MAX_SPREAD_PCT": 0.1,
            "MAX_OPEN_POSITIONS": 5,
            "MAX_POSITION_SIZE": 10000.0,
            "MAX_DAILY_LOSS": 1000.0,
            "LEVERAGE": 10,
            "RISK_PCT": 0.01,
            "TP_PERCENT": 2.0,
            "SL_PERCENT": 1.5,
            "NOTIFICATION_ENABLED": True,
            "ML_ENABLED": True,
            "SYMBOLS": ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "AVAXUSDT"],
            "EXCHANGE": "binance",
            "DISCORD_WEBHOOK_URL": "",
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
            "WHATSAPP_TO": ""
        }
        try:
            if save_settings(default_settings, user_id=user_id):
                try:
                    with db_manager.get_session() as session:
                        wallet = session.query(WalletBalance).filter_by(
                            user_id=user_id,
                            account_type="virtual",
                            exchange=current_exchange
                        ).first()
                        if wallet:
                            wallet.available = default_settings["VIRTUAL_BALANCE"]
                            wallet.total = default_settings["VIRTUAL_BALANCE"]
                            session.commit()
                except Exception as e:
                    logger.error(f"Error resetting virtual balance: {e}")
                    st.error(f"Error resetting virtual balance: {e}")
                st.success("Settings reset to defaults!")
                st.rerun()
            else:
                st.error("Failed to reset settings")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            st.error(f"Error: {e}")

with col3:
    if st.button("📤 Export Settings", use_container_width=True):
        try:
            settings_json = json.dumps(current_settings, indent=2)
            st.download_button(
                "💾 Download",
                data=settings_json,
                file_name=f"settings_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            st.error(f"Error: {e}")

# Status footer
st.divider()
st.markdown(f"""
**Status:** Exchange: {current_exchange.title()} | 
Mode: {account_type.title()} | 
User: {st.session_state.user.get('username', 'N/A')} | 
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""")