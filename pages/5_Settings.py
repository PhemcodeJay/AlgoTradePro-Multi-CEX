# pages/5_Settings.py
import streamlit as st
import os
import json
from datetime import datetime, timezone
import bcrypt

from db import db_manager, User, WalletBalance
from settings import load_settings, save_settings, validate_env
from logging_config import get_trading_logger
from multi_trading_engine import TradingEngine

logger = get_trading_logger(__name__)

# --------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------
st.set_page_config(
    page_title="Settings - AlgoTraderPro V2.0",
    page_icon="Settings",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------
# Session-state defaults
# --------------------------------------------------------------
defaults = {
    "authenticated": False,
    "user": None,
    "current_exchange": os.getenv("EXCHANGE", "bybit").lower(),
    "account_type": "virtual",
    "last_updated": "N/A",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------------------
# Header
# --------------------------------------------------------------
st.markdown(
    f"""
<div style='padding:1.5rem;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:10px;margin-bottom:1.5rem;'>
    <h1 style='color:white;margin:0;'>Settings System Settings</h1>
    <p style='color:white;margin:0.5rem 0 0 0;'>
        Configure trading parameters for <strong>{st.session_state.user.get('username','Guest') if st.session_state.user else 'N/A'}</strong>
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# Authentication helpers
# --------------------------------------------------------------
def authenticate_user(username: str, password: str) -> bool:
    try:
        with db_manager.get_session() as s:
            u = s.query(User).filter_by(username=username).first()
            if u and bcrypt.checkpw(password.encode(), u.password_hash.encode()):
                st.session_state.user = {"id": u.id, "username": u.username}
                st.session_state.authenticated = True
                logger.info(f"User {username} logged in")
                return True
    except Exception as e:
        logger.error(f"Login error: {e}")
    return False


def register_user(username: str, password: str) -> bool:
    try:
        with db_manager.get_session() as s:
            if s.query(User).filter_by(username=username).first():
                return False
            pwd = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            u = User(
                username=username,
                password_hash=pwd,
                binance_api_key=None,
                binance_api_secret=None,
                bybit_api_key=None,
                bybit_api_secret=None,
                default_exchange=st.session_state.current_exchange,
                trading_mode="virtual",
            )
            s.add(u)
            s.commit()

            for mode, bal in [("virtual", 1000.0), ("real", 0.0)]:
                s.add(
                    WalletBalance(
                        user_id=u.id,
                        account_type=mode,
                        available=bal,
                        used=0.0,
                        total=bal,
                        currency="USDT",
                        exchange=st.session_state.current_exchange,
                    )
                )
            s.commit()
            logger.info(f"Registered user {username}")
            return True
    except Exception as e:
        logger.error(f"Register error: {e}")
        return False


# --------------------------------------------------------------
# Login / Register UI
# --------------------------------------------------------------
if not st.session_state.authenticated:
    st.markdown("### Access Settings")
    t1, t2 = st.tabs(["Login", "Register"])

    with t1:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary"):
                if u and p and authenticate_user(u, p):
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    with t2:
        with st.form("register_form"):
            nu = st.text_input("Username")
            np1 = st.text_input("Password", type="password")
            np2 = st.text_input("Confirm", type="password")
            if st.form_submit_button("Register"):
                if not all([nu, np1, np2]):
                    st.error("Fill all fields")
                elif np1 != np2:
                    st.error("Passwords do not match")
                elif len(np1) < 6:
                    st.error("Password too short")
                elif register_user(nu, np1):
                    st.success("Registered – please login")
                else:
                    st.error("Username taken")
    st.stop()

# --------------------------------------------------------------
# Engine & user checks
# --------------------------------------------------------------
if "trading_engine" not in st.session_state or not st.session_state.trading_engine:
    st.error("Trading engine not initialized – go to the dashboard first.")
    if st.button("Go to Dashboard"):
        st.switch_page("app.py")
    st.stop()

trading_engine: TradingEngine = st.session_state.trading_engine
user_id = st.session_state.user["id"]
current_exchange = st.session_state.current_exchange
account_type = st.session_state.account_type

# --------------------------------------------------------------
# Load current settings + API keys from DB
# --------------------------------------------------------------
try:
    current_settings = load_settings(user_id=user_id) or {}
except Exception as e:
    logger.error(f"Settings load error: {e}")
    st.error("Failed to load settings")
    current_settings = {}

with db_manager.get_session() as s:
    db_user = s.query(User).get(user_id)
    binance_key = db_user.binance_api_key or ""
    binance_secret = db_user.binance_api_secret or ""
    bybit_key = db_user.bybit_api_key or ""
    bybit_secret = db_user.bybit_api_secret or ""

# --------------------------------------------------------------
# Quick-info cards
# --------------------------------------------------------------
st.markdown("### Quick Information")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Exchange", current_exchange.title())
with c2:
    st.metric("Mode", account_type.title())
with c3:
    api_ok = validate_env(current_exchange, user_id=user_id)[0]
    st.metric("API", "Connected" if api_ok else "Virtual Only")
with c4:
    bal = db_manager.get_wallet_balance(account_type, user_id, current_exchange)
    avail = bal.get("available", 1000.0 if account_type == "virtual" else 0.0)
    st.metric("Balance", f"${avail:,.2f}")

st.divider()

# --------------------------------------------------------------
# Tabs
# --------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Trading", "Exchange", "Risk Management", "Notifications", "System"]
)

# ---------------------- TAB 1: TRADING ----------------------
with tab1:
    st.markdown("### Trading Configuration")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**General**")
        scan_interval = st.number_input(
            "Scan Interval (s)",
            min_value=300.0,
            max_value=86400.0,
            value=float(current_settings.get("SCAN_INTERVAL", 3600.0)),
            step=300.0,
            format="%.0f"
        )
        top_n_signals = st.number_input(
            "Top N Signals",
            min_value=1,
            max_value=50,
            value=int(current_settings.get("TOP_N_SIGNALS", 10)),
            step=1
        )
        min_signal_score = st.slider(
            "Min Signal Score",
            min_value=0.0,
            max_value=100.0,
            value=float(current_settings.get("MIN_SIGNAL_SCORE", 40.0)),
            step=1.0
        )
        auto_trading = st.checkbox(
            "Auto Trading", value=current_settings.get("AUTO_TRADING_ENABLED", False)
        )
        virtual_balance = st.number_input(
            "Virtual Balance ($)",
            min_value=10.0,
            max_value=1_000_000.0,
            value=float(current_settings.get("VIRTUAL_BALANCE", 1000.0)),
            step=100.0,
            format="%.2f"
        )

    with col2:
        st.markdown("**Technical**")
        rsi_oversold = st.number_input(
            "RSI Oversold",
            min_value=10.0,
            max_value=40.0,
            value=float(current_settings.get("RSI_OVERSOLD", 30.0)),
            step=1.0,
            format="%.1f"
        )
        rsi_overbought = st.number_input(
            "RSI Overbought",
            min_value=60.0,
            max_value=90.0,
            value=float(current_settings.get("RSI_OVERBOUGHT", 70.0)),
            step=1.0,
            format="%.1f"
        )
        min_volume = st.number_input(
            "Min Volume (USDT)",
            min_value=100_000.0,
            max_value=10_000_000.0,
            value=float(current_settings.get("MIN_VOLUME", 1_000_000.0)),
            step=100_000.0,
            format="%.0f"
        )
        min_atr_pct = st.number_input(
            "Min ATR %",
            min_value=0.1,
            max_value=5.0,
            value=float(current_settings.get("MIN_ATR_PCT", 0.5)),
            step=0.1,
            format="%.2f"
        )
        max_spread_pct = st.number_input(
            "Max Spread %",
            min_value=0.01,
            max_value=1.0,
            value=float(current_settings.get("MAX_SPREAD_PCT", 0.1)),
            step=0.01,
            format="%.3f"
        )

# ---------------------- TAB 2: EXCHANGE ----------------------
with tab2:
    st.markdown("### Exchange Configuration")
    st.info("Tip: Switch exchanges on the main dashboard.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style='padding:1rem;background:#fff8e1;border-radius:8px;border:2px solid #ffd700;'>
                <h3 style='margin:0;'>Binance</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b_ok = validate_env("binance", user_id=user_id)[0]
        st.write("Configured" if b_ok else "Virtual Only")
    with col2:
        st.markdown(
            """
            <div style='padding:1rem;background:#fff3e0;border-radius:8px;border:2px solid #ff9800;'>
                <h3 style='margin:0;'>Bybit</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        by_ok = validate_env("bybit", user_id=user_id)[0]
        st.write("Configured" if by_ok else "Virtual Only")

    with st.expander("API Credentials", expanded=True):
        st.markdown("**Binance**")
        b_key_in = st.text_input(
            "Binance API Key",
            value=binance_key,
            type="password",
            key="b_key",
        )
        b_sec_in = st.text_input(
            "Binance API Secret",
            value=binance_secret,
            type="password",
            key="b_sec",
        )
        if st.button("Save Binance Keys", type="primary"):
            if b_key_in and b_sec_in:
                with db_manager.get_session() as s:
                    u = s.query(User).get(user_id)
                    u.binance_api_key = b_key_in
                    u.binance_api_secret = b_sec_in
                    s.commit()
                trading_engine.update_api_credentials("binance", b_key_in, b_sec_in)
                if current_exchange == "binance":
                    trading_engine.reload_settings()
                st.success("Binance keys saved & activated")
                st.rerun()
            else:
                st.error("Both fields required")

        st.markdown("**Bybit**")
        by_key_in = st.text_input(
            "Bybit API Key", value=bybit_key, type="password", key="by_key"
        )
        by_sec_in = st.text_input(
            "Bybit API Secret", value=bybit_secret, type="password", key="by_sec"
        )
        if st.button("Save Bybit Keys", type="primary"):
            if by_key_in and by_sec_in:
                with db_manager.get_session() as s:
                    u = s.query(User).get(user_id)
                    u.bybit_api_key = by_key_in
                    u.bybit_api_secret = by_sec_in
                    s.commit()
                trading_engine.update_api_credentials("bybit", by_key_in, by_sec_in)
                if current_exchange == "bybit":
                    trading_engine.reload_settings()
                st.success("Bybit keys saved & activated")
                st.rerun()
            else:
                st.error("Both fields required")

# ---------------------- TAB 3: RISK ----------------------
with tab3:
    st.markdown("### Risk Management")
    col1, col2 = st.columns(2)
    with col1:
        max_positions = st.number_input(
            "Max Open Positions",
            min_value=1,
            max_value=50,
            value=int(current_settings.get("MAX_OPEN_POSITIONS", 10)),
            step=1
        )
        max_position_size = st.number_input(
            "Max Position Size ($)",
            min_value=10.0,
            max_value=100_000.0,
            value=float(current_settings.get("MAX_POSITION_SIZE", 10_000.0)),
            step=100.0,
            format="%.2f"
        )
        max_daily_loss = st.number_input(
            "Max Daily Loss ($)",
            min_value=10.0,
            max_value=10_000.0,
            value=float(current_settings.get("MAX_DAILY_LOSS", 1_000.0)),
            step=50.0,
            format="%.2f"
        )
    with col2:
        leverage = st.number_input(
            "Leverage",
            min_value=1.0,
            max_value=20.0,
            value=float(current_settings.get("LEVERAGE", 10.0)),
            step=0.5,
            format="%.1f"
        )
        risk_pct = st.slider(
            "Risk % per Trade",
            min_value=0.1,
            max_value=10.0,
            value=float(current_settings.get("RISK_PCT", 0.02)) * 100,
            step=0.1
        )
        tp_percent = st.number_input(
            "Take-Profit %",
            min_value=0.5,
            max_value=20.0,
            value=float(current_settings.get("TP_PERCENT", 2.0)),
            step=0.1,
            format="%.2f"
        )
        sl_percent = st.number_input(
            "Stop-Loss %",
            min_value=0.5,
            max_value=20.0,
            value=float(current_settings.get("SL_PERCENT", 1.5)),
            step=0.1,
            format="%.2f"
        )

# ---------------------- TAB 4: NOTIFICATIONS ----------------------
with tab4:
    st.markdown("### Notification Settings")
    notif_enabled = st.checkbox(
        "Enable Notifications", value=current_settings.get("NOTIFICATION_ENABLED", True)
    )
    discord = st.text_input(
        "Discord Webhook",
        value=current_settings.get("DISCORD_WEBHOOK_URL", ""),
        type="password",
    )
    tg_token = st.text_input(
        "Telegram Bot Token",
        value=current_settings.get("TELEGRAM_BOT_TOKEN", ""),
        type="password",
    )
    tg_chat = st.text_input(
        "Telegram Chat ID", value=current_settings.get("TELEGRAM_CHAT_ID", "")
    )
    wa = st.text_input(
        "WhatsApp Number", value=current_settings.get("WHATSAPP_TO", "")
    )

# ---------------------- TAB 5: SYSTEM ----------------------
with tab5:
    st.markdown("### System Configuration")
    col1, col2 = st.columns(2)
    with col1:
        symbols_txt = st.text_area(
            "Symbols (one per line)",
            value="\n".join(current_settings.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])),
            height=200,
        )
        ml_enabled = st.checkbox(
            "ML Filtering", value=current_settings.get("ML_ENABLED", True)
        )
    with col2:
        st.markdown("**Status**")
        st.write(f"Exchange: {current_exchange.title()}")
        st.write(
            f"Trading: {'ON' if trading_engine.is_trading_enabled() else 'OFF'}"
        )
        st.write(f"Last Save: {st.session_state.last_updated}")

# --------------------------------------------------------------
# Save / Reset / Export
# --------------------------------------------------------------
st.divider()
st.markdown("### Save Configuration")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Save All Settings", type="primary", use_container_width=True):
        new_settings = {
            "SCAN_INTERVAL": float(scan_interval),
            "TOP_N_SIGNALS": int(top_n_signals),
            "MIN_SIGNAL_SCORE": float(min_signal_score),
            "AUTO_TRADING_ENABLED": bool(auto_trading),
            "VIRTUAL_BALANCE": float(virtual_balance),
            "RSI_OVERSOLD": float(rsi_oversold),
            "RSI_OVERBOUGHT": float(rsi_overbought),
            "MIN_VOLUME": float(min_volume),
            "MIN_ATR_PCT": float(min_atr_pct),
            "MAX_SPREAD_PCT": float(max_spread_pct),
            "MAX_OPEN_POSITIONS": int(max_positions),
            "MAX_POSITION_SIZE": float(max_position_size),
            "MAX_DAILY_LOSS": float(max_daily_loss),
            "LEVERAGE": float(leverage),
            "RISK_PCT": float(risk_pct / 100),
            "TP_PERCENT": float(tp_percent),
            "SL_PERCENT": float(sl_percent),
            "NOTIFICATION_ENABLED": bool(notif_enabled),
            "ML_ENABLED": bool(ml_enabled),
            "SYMBOLS": [s.strip().upper() for s in symbols_txt.split("\n") if s.strip()],
            "EXCHANGE": current_exchange,
            "DISCORD_WEBHOOK_URL": discord,
            "TELEGRAM_BOT_TOKEN": tg_token,
            "TELEGRAM_CHAT_ID": tg_chat,
            "WHATSAPP_TO": wa,
        }

        if not new_settings["SYMBOLS"]:
            st.error("At least one symbol required")
        else:
            if save_settings(new_settings, user_id=user_id):
                # Update virtual balance if changed
                if virtual_balance != current_settings.get("VIRTUAL_BALANCE", 1000):
                    with db_manager.get_session() as s:
                        w = (
                            s.query(WalletBalance)
                            .filter_by(
                                user_id=user_id,
                                account_type="virtual",
                                exchange=current_exchange,
                            )
                            .first()
                        )
                        if w:
                            w.available = w.total = virtual_balance
                            s.commit()
                trading_engine.reload_settings()
                st.session_state.last_updated = datetime.now(
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                st.success("Settings saved!")
                st.balloons()
            else:
                st.error("Save failed")

with c2:
    if st.button("Reset to Defaults", type="secondary", use_container_width=True):
        defaults = {
            "SCAN_INTERVAL": 3600.0,
            "TOP_N_SIGNALS": 10,
            "MIN_SIGNAL_SCORE": 40.0,
            "AUTO_TRADING_ENABLED": False,
            "VIRTUAL_BALANCE": 1000.0,
            "RSI_OVERSOLD": 30.0,
            "RSI_OVERBOUGHT": 70.0,
            "MIN_VOLUME": 1_000_000.0,
            "MIN_ATR_PCT": 0.5,
            "MAX_SPREAD_PCT": 0.1,
            "MAX_OPEN_POSITIONS": 10,
            "MAX_POSITION_SIZE": 10_000.0,
            "MAX_DAILY_LOSS": 1_000.0,
            "LEVERAGE": 10.0,
            "RISK_PCT": 0.02,
            "TP_PERCENT": 2.0,
            "SL_PERCENT": 1.5,
            "NOTIFICATION_ENABLED": True,
            "ML_ENABLED": True,
            "SYMBOLS": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "EXCHANGE": "bybit",
            "DISCORD_WEBHOOK_URL": "",
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_CHAT_ID": "",
            "WHATSAPP_TO": "",
        }
        if save_settings(defaults, user_id=user_id):
            with db_manager.get_session() as s:
                u = s.query(User).get(user_id)
                u.binance_api_key = u.binance_api_secret = None
                u.bybit_api_key = u.bybit_api_secret = None
                s.commit()
                w = s.query(WalletBalance).filter_by(
                    user_id=user_id, account_type="virtual"
                ).first()
                if w:
                    w.available = w.total = 1000.0
                    s.commit()
            trading_engine.reload_settings()
            st.session_state.last_updated = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.success("Reset to defaults!")
            st.rerun()

with c3:
    if st.button("Export Settings", type="secondary", use_container_width=True):
        data = json.dumps(current_settings, indent=2)
        st.download_button(
            "Download JSON",
            data,
            f"settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
        )

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.divider()
st.caption(
    f"User: {st.session_state.user['username']} | Last Save: {st.session_state.last_updated}"
)