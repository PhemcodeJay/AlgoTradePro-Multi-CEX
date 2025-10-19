import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from db import DatabaseManager, User
from logging_config import get_trading_logger
from utils import format_price
import bcrypt

# Configure page
st.set_page_config(
    page_title="Signals - AlgoTraderPro V2.0",
    page_icon="📡",
    layout="wide"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Helper functions for authentication (same as app.py and 1_Dashboard.py)
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

            from db import WalletBalance
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

    st.stop()  # Stop rendering until authenticated

# Initialize components
if 'trading_engine' not in st.session_state or st.session_state.trading_engine is None:
    st.error("Trading engine not initialized. Please return to the main page to initialize the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
db_manager = DatabaseManager()
current_exchange = st.session_state.get('current_exchange', 'binance')
account_type = st.session_state.get('account_type', 'virtual')
user_id = st.session_state.user.get('id') if st.session_state.user else None

if not user_id:
    st.error("User not authenticated. Please log in from the main page.")
    st.stop()

# Main content
st.title("📡 Trading Signals")
st.markdown(f"**Exchange:** {current_exchange.title()} | **Mode:** {account_type.title()}")

# Signal Filters
st.subheader("🔍 Filter Signals")
col1, col2, col3 = st.columns(3)
with col1:
    symbol_filter = st.text_input("Symbol (e.g., BTCUSDT)", "").upper()
with col2:
    side_filter = st.selectbox("Side", ["All", "Buy", "Sell", "Long", "Short"])
with col3:
    limit = st.selectbox("Number of Signals", [10, 25, 50, 100], index=1)

# Fetch and display signals
try:
    signals = db_manager.get_signals(
        limit=limit,
        exchange=current_exchange,
        user_id=user_id,
        symbol=symbol_filter if symbol_filter else None,
        side=side_filter if side_filter != "All" else None
    )
    if signals:
        signal_data = []
        for s in signals:
            created_at = datetime.fromisoformat(s['created_at']) if s.get('created_at') else None
            signal_data.append({
                'Symbol': s.get('symbol', 'N/A'),
                'Side': s.get('side', 'N/A'),
                'Score': f"{s.get('score', 0):.1f}",
                'Entry': format_price(s.get('entry', 0), 6),
                'Market': s.get('market', 'N/A'),
                'Created': created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'N/A'
            })
        df_signals = pd.DataFrame(signal_data)

        def color_side(val):
            if val.upper() in ['BUY', 'LONG']:
                return 'color: green'
            elif val.upper() in ['SELL', 'SHORT']:
                return 'color: red'
            return 'color: gray'

        styled_df = df_signals.style.applymap(color_side, subset=['Side'])
        st.dataframe(styled_df, use_container_width=True, height=500)
    else:
        st.info("No signals found for the selected filters.")
except Exception as e:
    logger.error(f"Error loading signals: {e}")
    st.error(f"Error loading signals: {e}")

# Status footer
st.markdown(f"""
---
**Status:** Trading Engine: {current_exchange.title()} | 
Trading: {'🟢 Active' if hasattr(trading_engine, 'is_trading_enabled') and trading_engine.is_trading_enabled() else '🔴 Inactive'} | 
Mode: {account_type.title()} | 
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""")