import streamlit as st
import os
from datetime import datetime, timezone
import bcrypt

# Core imports
from db import db_manager, User, WalletBalance
from settings import load_settings, validate_env, save_settings
from logging_config import get_trading_logger
from multi_trading_engine import TradingEngine
from automated_trader import AutomatedTrader

# ========================= CONFIG =========================
st.set_page_config(
    page_title="AlgoTraderPro - MultiEX",
    page_icon="Chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger = get_trading_logger(__name__)

# ========================= SESSION STATE =========================
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': False,
        'trading_engine': None,
        'automated_trader': None,
        'current_exchange': os.getenv("EXCHANGE", "bybit").lower(),
        'trading_active': False,
        'account_type': 'virtual',
        'authenticated': False,
        'user': None,
        'has_api_keys': False,
        'just_logged_in': False,
    })

def to_float(val):
    try:
        return float(val or 0)
    except:
        return 0.0


# ========================= AUTH =========================
def authenticate_user(username: str, password: str) -> bool:
    try:
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
                st.session_state.user = {"id": user.id, "username": user.username}
                st.session_state.authenticated = True
                st.session_state.just_logged_in = True
                return True
            return False
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return False


def register_user(username: str, password: str) -> bool:
    if not username.strip() or len(password) < 6:
        return False

    try:
        with db_manager.get_session() as session:
            if session.query(User).filter_by(username=username).first():
                return False

            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            new_user = User(username=username, password_hash=hashed, default_exchange="bybit")
            session.add(new_user)
            session.flush()

            for acc in ("virtual", "real"):
                session.add(WalletBalance(
                    user_id=new_user.id,
                    account_type=acc,
                    available=100.0 if acc == "virtual" else 0.0,
                    used=0.0,
                    total=100.0 if acc == "virtual" else 0.0,
                    currency="USDT",
                    exchange="bybit",
                ))

            save_settings({"EXCHANGE": "bybit", "TRADING_MODE": "virtual"}, new_user.id)
            session.commit()
            logger.info(f"Registered: {username}")
            return True
    except Exception as e:
        logger.error(f"Register failed: {e}", exc_info=True)
        try:
            session.rollback()
        except:
            pass
        return False


# ========================= LOGIN UI =========================
if not st.session_state.authenticated:
    st.title("AlgoTraderPro - Kenya's #1 Trading Bot")
    st.markdown("### Login or Create Account (Free Forever)")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Login")
        lu = st.text_input("Username", key="lu", placeholder="Enter username")
        lp = st.text_input("Password", type="password", key="lp", placeholder="Enter password")
        if st.button("Login", type="primary", use_container_width=True):
            if lu and lp:
                with st.spinner("Logging in..."):
                    if authenticate_user(lu, lp):
                        st.rerun()
                    else:
                        st.error("Wrong credentials")
            else:
                st.error("Fill all fields")

    with c2:
        st.subheader("Register")
        ru = st.text_input("Username", key="ru", placeholder="Choose username")
        rp1 = st.text_input("Password", type="password", key="rp1", placeholder="6+ chars")
        rp2 = st.text_input("Confirm", type="password", key="rp2")
        if st.button("Create Account", use_container_width=True):
            if not ru.strip():
                st.error("Username required")
            elif len(rp1) < 6:
                st.error("Password too short")
            elif rp1 != rp2:
                st.error("Passwords don't match")
            else:
                with st.spinner("Creating..."):
                    if register_user(ru, rp1):
                        st.success("Done! Logging in...")
                        authenticate_user(ru, rp1)
                        st.rerun()
                    else:
                        st.error("Username taken")

    st.stop()


# Clear cache after login
if st.session_state.just_logged_in:
    st.cache_data.clear()
    st.session_state.just_logged_in = False
    st.rerun()


# ========================= INIT ENGINE =========================
def initialize_app():
    try:
        uid = st.session_state.user["id"]
        settings = load_settings(uid) or {}
        ex = settings.get("EXCHANGE", "bybit").lower()
        mode = settings.get("TRADING_MODE", "virtual").lower()

        has_keys, _ = validate_env(exchange=ex, user_id=uid)
        st.session_state.has_api_keys = has_keys
        st.session_state.account_type = "virtual" if not has_keys else mode

        engine = TradingEngine(uid, ex, st.session_state.account_type)
        engine.switch_exchange(ex)
        if st.session_state.account_type == "real" and has_keys:
            engine.enable_trading()

        trader = AutomatedTrader(engine)

        st.session_state.trading_engine = engine
        st.session_state.automated_trader = trader
        st.session_state.current_exchange = ex
        st.session_state.initialized = True

        return True
    except Exception as e:
        logger.error(f"Init failed: {e}", exc_info=True)
        st.error("Bot failed to start")
        return False


if not st.session_state.initialized:
    if not initialize_app():
        st.stop()
    st.session_state.automated_trader.start()
    st.success("Bot is LIVE!")


# ========================= HEADER =========================
st.markdown(f"""
<div style='padding: 1.5rem; background: linear-gradient(90deg, #00C9FF, #92FE9D); 
     border-radius: 15px; text-align: center; color: black; margin-bottom: 2rem;'>
    <h1 style='margin:0;'>AlgoTraderPro Dashboard</h1>
    <p style='margin:5px 0 0; font-size:1.2rem;'>
        <strong>{st.session_state.user['username']}</strong> • 
        {st.session_state.current_exchange.title()} • 
        {st.session_state.account_type.title()} Mode
    </p>
</div>
""", unsafe_allow_html=True)


# ========================= SIDEBAR =========================
st.sidebar.header("Controls")

# Exchange
ex_opts = ["Bybit", "Binance"]
ex_idx = 0 if st.session_state.current_exchange == "bybit" else 1
sel_ex = st.sidebar.selectbox("Exchange", ex_opts, index=ex_idx, key="ex")
if sel_ex.lower() != st.session_state.current_exchange:
    st.session_state.current_exchange = sel_ex.lower()
    st.session_state.trading_engine.switch_exchange(sel_ex.lower())
    s = load_settings(st.session_state.user["id"])
    s["EXCHANGE"] = sel_ex.lower()
    save_settings(s, st.session_state.user["id"])
    st.cache_data.clear()
    st.rerun()

# Mode
mode_opts = ["Virtual", "Real"] if st.session_state.has_api_keys else ["Virtual"]
mode_idx = 0 if st.session_state.account_type == "virtual" else 1
sel_mode = st.sidebar.selectbox("Mode", mode_opts, index=mode_idx, key="mode")
if sel_mode.lower() != st.session_state.account_type:
    st.session_state.account_type = sel_mode.lower()
    st.session_state.trading_engine.set_account_type(sel_mode.lower())
    s = load_settings(st.session_state.user["id"])
    s["TRADING_MODE"] = sel_mode.lower()
    save_settings(s, st.session_state.user["id"])
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Trading")

enabled = st.session_state.trading_engine.is_trading_enabled()
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.sidebar.button("START", disabled=enabled, use_container_width=True):
        st.session_state.trading_engine.enable_trading()
        st.rerun()
with c2:
    if st.sidebar.button("STOP", disabled=not enabled, use_container_width=True):
        st.session_state.trading_engine.disable_trading("UI Stop")
        st.rerun()

if st.sidebar.button("EMERGENCY STOP", type="primary", use_container_width=True):
    st.session_state.trading_engine.emergency_stop("USER EMERGENCY")
    st.error("EMERGENCY STOP ACTIVATED")
    st.rerun()

st.sidebar.markdown(f"**Status:** {'LIVE' if enabled else 'OFF'}")


# ========================= DATA (FIXED) =========================
@st.cache_data(ttl=20)
def get_data(exchange, acc_type, is_virtual):
    uid = st.session_state.user["id"]
    with db_manager.get_session() as s:
        signals = db_manager.get_signals(10, exchange, uid) or []
        wallet = db_manager.get_wallet_balance(acc_type, uid, exchange) or {}
        trades = db_manager.get_trades(1000, is_virtual, exchange, uid) or []

        open_t = [t for t in trades if getattr(t, 'status', '') == 'open']
        closed_t = [t for t in trades if getattr(t, 'status', '') == 'closed']

        def to_dict(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return obj
            else:
                return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}

        return {
            "signals": [to_dict(s) for s in signals],
            "wallet": wallet if isinstance(wallet, dict) else (wallet.to_dict() if wallet else {"available": 100 if is_virtual else 0}),
            "open": [to_dict(t) for t in open_t],
            "closed": [to_dict(t) for t in closed_t],
        }

v = get_data(st.session_state.current_exchange, "virtual", True)
r = get_data(st.session_state.current_exchange, "real", False)


# ========================= DASHBOARD =========================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Virtual Balance", f"${v['wallet'].get('available', 100):.2f}")
with col2:
    st.metric("Real Balance", f"${r['wallet'].get('available', 0):.2f}")
with col3:
    st.metric("Open Virtual", len(v["open"]))
with col4:
    st.metric("Open Real", len(r["open"]))

st.divider()

t1, t2, t3 = st.tabs(["Signals", "Virtual Trades", "Real Trades"])

with t1:
    for s in v["signals"][:5]:
        st.write(f"**{s['symbol']}** → {s['side'].upper()} | ${s['entry']:.6f}")

with t2:
    for t in v["closed"][-5:]:
        color = "green" if t["pnl"] > 0 else "red"
        st.markdown(f":{color}[{t['symbol']} • ${t['pnl']:.2f}]")

with t3:
    for t in r["closed"][-5:]:
        color = "green" if t["pnl"] > 0 else "red"
        st.markdown(f":{color}[{t['symbol']} • ${t['pnl']:.2f}]")


# ========================= FOOTER =========================
if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>AlgoTraderPro v2.0</strong> • Kenya • 
    {datetime.now(timezone.utc).strftime('%b %d, %Y %I:%M %p')} UTC
    <br><small>Built for Kenyan traders | Running 24/7</small>
</div>
""", unsafe_allow_html=True)