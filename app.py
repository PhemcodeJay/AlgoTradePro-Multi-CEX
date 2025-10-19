import streamlit as st
import os
import sys
from datetime import datetime, timezone
import asyncio
import threading
import time
from typing import Dict, Any, List
import bcrypt

# Import core modules
from db import db_manager, User, WalletBalance
from settings import load_settings, validate_env, save_settings
from logging_config import get_trading_logger
from multi_trading_engine import TradingEngine
from automated_trader import AutomatedTrader

# Configure page
st.set_page_config(
    page_title="AlgoTraderPro - MultiEX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.trading_engine = None
    st.session_state.automated_trader = None
    st.session_state.current_exchange = os.getenv("EXCHANGE", "binance").lower()
    st.session_state.trading_active = False
    st.session_state.account_type = 'virtual'  # Default to virtual
    st.session_state.authenticated = False  # Track authentication state
    st.session_state.user = None  # Store user info (dict with id, username)

def to_float(value):
    try:
        return float(value or 0)
    except Exception:
        return 0.0

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
            # Check if username already exists
            if session.query(User).filter_by(username=username).first():
                logger.warning(f"Registration failed: Username {username} already exists")
                return False
            
            # Hash password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Create new user
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

            # Initialize default wallet balances
            session.add(WalletBalance(
                user_id=new_user.id,
                account_type="virtual",
                available=100.0,
                used=0.0,
                total=100.0,
                currency="USDT",
                exchange=st.session_state.current_exchange
            ))
            session.add(WalletBalance(
                user_id=new_user.id,
                account_type="real",
                available=0.0,
                used=0.0,
                total=0.0,
                currency="USDT",
                exchange=st.session_state.current_exchange
            ))
            session.commit()
            logger.info(f"Registered new user: {username}, ID: {new_user.id}")
            return True
    except Exception as e:
        logger.error(f"Registration error for {username}: {e}")
        session.rollback()
        return False

def initialize_app():
    """Initialize the application components"""
    try:
        # Load settings
        settings = load_settings()
        exchange = settings.get("EXCHANGE", "binance").lower()
        trading_mode = settings.get("TRADING_MODE", "virtual").lower()

        # Check if API keys are available
        has_api_keys = validate_env(exchange)

        if not bool(has_api_keys):
            st.warning(f"⚠️ No API credentials found for {exchange.title()}. Running in **Virtual Mode Only**.")

            with st.expander("📝 About Virtual Mode", expanded=False):
                st.markdown(f"""
                ### Virtual Trading Mode

                You're currently running in **Virtual Mode** which allows you to:
                - ✅ Fetch real market data
                - ✅ Generate trading signals
                - ✅ Execute virtual trades
                - ✅ Track performance
                - ❌ Cannot execute real trades

                ### To Enable Real Trading:

                1. Click on the **Secrets** tool (🔒) in the left sidebar
                2. Add your exchange API credentials:

                **For Binance:**
                - `BINANCE_API_KEY` = your Binance API key
                - `BINANCE_API_SECRET` = your Binance API secret

                **For Bybit:**
                - `BYBIT_API_KEY` = your Bybit API key
                - `BYBIT_API_SECRET` = your Bybit API secret

                3. Click **Save** and restart the application
                """)

            # Force virtual mode
            st.session_state.account_type = 'virtual'
        else:
            st.session_state.account_type = trading_mode

        # Initialize trading engine
        trading_engine: TradingEngine = TradingEngine()
        trading_engine.switch_exchange(exchange)
        if trading_mode == 'real' and bool(has_api_keys):
            trading_engine.enable_trading()

        # Initialize automated trader
        automated_trader = AutomatedTrader(trading_engine)

        # Store in session state
        st.session_state.trading_engine = trading_engine
        st.session_state.automated_trader = automated_trader
        st.session_state.current_exchange = exchange
        st.session_state.initialized = True
        logger.info(f"App initialized successfully: exchange={exchange}, trading_mode={trading_mode}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}")
        st.error(f"Application initialization failed: {str(e)}")
        return False

@st.cache_data
def get_cached_dashboard_data(exchange: str, account_type: str, _is_virtual: bool) -> Dict[str, Any]:
    """Fetch and cache dashboard data as plain dictionaries"""
    try:
        # Ensure user_id is available
        user_id = st.session_state.user.get('id') if st.session_state.user else None
        if not user_id:
            logger.error("No user_id found in session state for dashboard data")
            return {
                'signals': [],
                'wallet': {'available': 0.0},
                'open_trades': [],
                'closed_trades': []
            }

        with db_manager.get_session() as session:
            # Fetch signals
            signals = db_manager.get_signals(limit=10, exchange=exchange, user_id=user_id) or []
            signals_data = [s.to_dict() for s in signals]

            # Fetch wallet balance
            wallet = db_manager.get_wallet_balance(account_type=account_type, user_id=user_id, exchange=exchange)
            wallet_data = wallet.to_dict() if wallet else {'available': 0.0, 'used': 0.0, 'total': 0.0}

            # Fetch trades
            open_trades = db_manager.get_trades(status='open', virtual=_is_virtual, exchange=exchange, user_id=user_id) or []
            closed_trades = db_manager.get_trades(status='closed', virtual=_is_virtual, exchange=exchange, user_id=user_id) or []
            open_trades_data = [t.to_dict() for t in open_trades]
            closed_trades_data = [t.to_dict() for t in closed_trades]

            return {
                'signals': signals_data,
                'wallet': wallet_data,
                'open_trades': open_trades_data,
                'closed_trades': closed_trades_data
            }
    except Exception as e:
        logger.error(f"Error fetching cached dashboard data: {e}")
        return {
            'signals': [],
            'wallet': {'available': 0.0},
            'open_trades': [],
            'closed_trades': []
        }

@st.cache_data
def get_cached_signals(exchange: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch and cache recent signals as plain dictionaries"""
    try:
        user_id = st.session_state.user.get('id') if st.session_state.user else None
        if not user_id:
            logger.error("No user_id found in session state for signals")
            return []
        with db_manager.get_session() as session:
            signals = db_manager.get_signals(limit=limit, exchange=exchange, user_id=user_id) or []
            return [s.to_dict() for s in signals]
    except Exception as e:
        logger.error(f"Error fetching cached signals: {e}")
        return []

@st.cache_data
def get_cached_trades(exchange: str, _is_virtual: bool, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch and cache recent trades as plain dictionaries"""
    try:
        user_id = st.session_state.user.get('id') if st.session_state.user else None
        if not user_id:
            logger.error("No user_id found in session state for trades")
            return []
        with db_manager.get_session() as session:
            trades = db_manager.get_trades(status='closed', virtual=_is_virtual, exchange=exchange, limit=limit, user_id=user_id) or []
            return [t.to_dict() for t in trades]
    except Exception as e:
        logger.error(f"Error fetching cached trades: {e}")
        return []

def main():
    """Main application entry point"""
    try:
        # Application header
        st.title("🚀 AlgoTraderPro V2.0")
        st.markdown("*Advanced Multi-Exchange Cryptocurrency Trading Platform*")

        # Login/Register Section on Main Page
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

        # Initialize app if not done
        if not bool(st.session_state.initialized):
            with st.spinner("Initializing AlgoTrader Pro..."):
                if not initialize_app():
                    st.stop()

        # Welcome message for authenticated user
        st.markdown(f"👋 Welcome, {st.session_state.user['username']}!")

    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        st.error("⚠️ Application Error - Please refresh the page")
        st.stop()

    # Load settings for current exchange and mode
    settings = load_settings()
    current_exchange = st.session_state.get('current_exchange', settings.get("EXCHANGE", "binance")).lower()
    account_type = st.session_state.get('account_type', settings.get("TRADING_MODE", "virtual")).lower()

    # Exchange Selection at Top
    st.markdown("### 🏦 Select Exchange")
    col1, col2 = st.columns(2)

    with col1:
        binance_type = "primary" if current_exchange == "binance" else "secondary"
        if st.button("🟡 Binance", type=binance_type, use_container_width=True, key="exchange_binance"):
            if current_exchange != "binance":
                st.session_state.current_exchange = "binance"
                try:
                    engine = st.session_state.trading_engine
                    if engine is not None and bool(engine.switch_exchange("binance")):
                        st.success("✅ Switched to Binance")
                        settings["EXCHANGE"] = "binance"
                        save_settings(settings)
                        st.rerun()
                    else:
                        st.error("Failed to switch to Binance")
                except Exception as e:
                    st.error(f"Error switching to Binance: {e}")

    with col2:
        bybit_type = "primary" if current_exchange == "bybit" else "secondary"
        if st.button("🟠 Bybit", type=bybit_type, use_container_width=True, key="exchange_bybit"):
            if current_exchange != "bybit":
                st.session_state.current_exchange = "bybit"
                try:
                    engine = st.session_state.trading_engine
                    if engine is not None and bool(engine.switch_exchange("bybit")):
                        st.success("✅ Switched to Bybit")
                        settings["EXCHANGE"] = "bybit"
                        save_settings(settings)
                        st.rerun()
                    else:
                        st.error("Failed to switch to Bybit")
                except Exception as e:
                    st.error(f"Error switching to Bybit: {e}")

    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")

        # Display selected exchange
        st.info(f"📍 **Active Exchange:** {current_exchange.title()}")

        # Account Type selection
        st.divider()
        st.subheader("💼 Trading Mode")

        col1, col2 = st.columns(2)

        with col1:
            virtual_type = "primary" if account_type == "virtual" else "secondary"
            if st.button("🎮 Virtual Trading", type=virtual_type, use_container_width=True, key="mode_virtual"):
                if account_type != "virtual":
                    st.session_state.account_type = "virtual"
                    settings["TRADING_MODE"] = "virtual"
                    save_settings(settings)
                    st.success("✅ Switched to Virtual Trading mode")
                    st.rerun()

        with col2:
            has_api_keys = validate_env(current_exchange, allow_virtual=False)
            real_type = "primary" if account_type == "real" else "secondary"
            has_api_keys_bool = bool(has_api_keys)
            if st.button("💰 Real Trading", type=real_type, disabled=not has_api_keys_bool, use_container_width=True, key="mode_real"):
                if account_type != "real":
                    if has_api_keys_bool:
                        st.session_state.account_type = "real"
                        settings["TRADING_MODE"] = "real"
                        save_settings(settings)
                        st.success("✅ Switched to Real Trading mode")
                        st.rerun()
                    else:
                        st.error(f"❌ Real trading requires {current_exchange.upper()} API credentials in Secrets")

        # Show current mode info
        if account_type == "virtual":
            st.info("ℹ️ Virtual mode uses simulated data and trades")
        else:
            st.warning("⚠️ Real mode uses live market data and executes actual trades")

        st.divider()

        # Trading controls
        st.subheader("🎯 Trading Status")

        if st.session_state.trading_engine is not None:
            trading_enabled = bool(st.session_state.trading_engine.is_trading_enabled())

            col1, col2 = st.columns(2)

            with col1:
                start_disabled = trading_enabled or (account_type == 'virtual')
                if st.button("▶️ Start Trading", disabled=start_disabled):
                    try:
                        if st.session_state.trading_engine is not None and bool(st.session_state.trading_engine.enable_trading()):
                            st.success("Trading enabled")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Error enabling trading: {e}")
                        st.error("Failed to enable trading")

            with col2:
                if st.button("⏸️ Stop Trading", disabled=not trading_enabled):
                    try:
                        if st.session_state.trading_engine is not None and bool(st.session_state.trading_engine.disable_trading("Manual stop")):
                            st.warning("Trading disabled")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Error disabling trading: {e}")
                        st.error("Failed to disable trading")

            # Emergency stop
            if st.button("🚨 Emergency Stop", type="secondary"):
                try:
                    if st.session_state.trading_engine is not None and bool(st.session_state.trading_engine.emergency_stop("Manual emergency stop")):
                        st.error("Emergency stop activated")
                        st.rerun()
                except Exception as e:
                    logger.error(f"Error during emergency stop: {e}")
                    st.error("Emergency stop failed")

            # Status indicators
            if trading_enabled:
                st.success("🟢 Trading Active")
            else:
                st.error("🔴 Trading Inactive")

        st.divider()

        # Quick stats
        st.subheader("📊 Quick Stats")

        if st.session_state.trading_engine is not None:
            try:
                user_id = st.session_state.user.get('id') if st.session_state.user else None
                if not user_id:
                    st.error("User ID not found. Please log in again.")
                    st.stop()

                wallet = db_manager.get_wallet_balance(account_type=account_type, user_id=user_id, exchange=current_exchange)
                stats = st.session_state.trading_engine.get_trade_statistics(account_type)

                if wallet is not None:
                    available_val = to_float(wallet.get('available'))
                    st.metric(f"{account_type.title()} Balance", f"${available_val:.2f}")
                else:
                    st.metric(f"{account_type.title()} Balance", f"$0.00")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Trades", int(stats.get('total_trades', 0)))
                with col2:
                    st.metric("Win Rate", f"{float(stats.get('win_rate', 0)):.1f}%")

                st.metric("Total P&L", f"${float(stats.get('total_pnl', 0)):.2f}")

            except Exception as e:
                logger.error(f"Error loading quick stats: {e}")
                st.error("Error loading stats")

    # Main content area with cards
    mode_emoji = "🧪" if account_type == 'virtual' else "💰"
    st.markdown(f"""
    <div style='padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>📈 Dashboard - {current_exchange.title()}</h2>
        <p style='color: white; margin: 0.5rem 0 0 0;'>{mode_emoji} {account_type.title()} Trading Mode</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row with cards
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    is_virtual = account_type == 'virtual'
    
    try:
        # Fetch cached dashboard data
        dashboard_data = get_cached_dashboard_data(current_exchange, account_type, is_virtual)
        signals = dashboard_data['signals']
        wallet = dashboard_data['wallet']
        open_trades = dashboard_data['open_trades']
        closed_trades = dashboard_data['closed_trades']

        # Calculate today's P&L
        today_date = datetime.now(timezone.utc).date().isoformat()
        today_trades = [t for t in closed_trades if t['created_at'].startswith(today_date)]
        today_pnl = sum(t['pnl'] for t in today_trades)

        with col1:
            recent_count = len(signals)
            last_scan = "Never"
            if recent_count > 0 and signals[0]['created_at'] != 'N/A':
                last_scan = signals[0]['created_at'].split(' ')[1][:5]  # Extract HH:MM
            st.metric("Recent Signals", recent_count, delta=f"Last scan: {last_scan}")

        with col2:
            open_count = len(open_trades)
            max_open = getattr(st.session_state.trading_engine, "max_open_positions", "N/A") if st.session_state.trading_engine is not None else "N/A"
            st.metric("Open Positions", open_count, delta=f"Max: {max_open}")

        with col3:
            balance_label = f"{account_type.title()} Balance"
            balance_value = f"${wallet['available']:.2f}"
            st.metric(balance_label, balance_value)

        with col4:
            pnl_value = float(to_float(today_pnl))
            pnl_delta = "➖"
            if pnl_value > 0:
                pnl_delta = "📈"
            elif pnl_value < 0:
                pnl_delta = "📉"
            st.metric("Today's P&L", f"${pnl_value:.2f}", delta=pnl_delta)

    except Exception as e:
        logger.error(f"Error loading dashboard metrics: {e}")
        st.error("Error loading dashboard data")

    # Recent activity
    st.subheader("🔄 Recent Activity")

    tab1, tab2 = st.tabs(["Latest Signals", "Recent Trades"])

    with tab1:
        try:
            signals = get_cached_signals(current_exchange, limit=5)
            if len(signals) > 0:
                for signal in signals:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.write(f"**{signal['symbol']}** - {signal['side']}")
                            st.caption(f"Created: {signal['created_at']}")
                        with col2:
                            st.metric("Score", f"{signal['score']:.1f}")
                        with col3:
                            st.metric("Entry", f"${signal['entry']:.6f}")
                        with col4:
                            side_upper = signal['side'].upper()
                            color = "green" if side_upper in ["BUY", "LONG"] else "red"
                            st.markdown(f":{color}[{signal['market']}]")
                        st.divider()
            else:
                st.info("No recent signals found")
        except Exception as e:
            logger.error(f"Error loading recent signals: {e}")
            st.error("Error loading signals")

    with tab2:
        try:
            recent_trades = get_cached_trades(current_exchange, is_virtual, limit=5)
            if len(recent_trades) > 0:
                for trade in reversed(recent_trades):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.write(f"**{trade['symbol']}** - {trade['side']}")
                            st.caption(f"Closed: {trade['updated_at']}")
                        with col2:
                            st.metric("Qty", f"{trade['qty']:.6f}")
                        with col3:
                            st.metric("P&L", f"${trade['pnl']:.2f}")
                        with col4:
                            color = "green" if trade['pnl'] > 0 else "red" if trade['pnl'] < 0 else "gray"
                            label = "Profit" if trade['pnl'] > 0 else "Loss" if trade['pnl'] < 0 else "Break Even"
                            st.markdown(f":{color}[{label}]")
                        st.divider()
            else:
                st.info("No recent trades found")
        except Exception as e:
            logger.error(f"Error loading recent trades: {e}")
            st.error("Error loading trades")

    # Auto-refresh
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()  # Clear cache on refresh
        st.rerun()

    # Footer
    st.divider()
    st.markdown(
        f"""
        <div style='text-align: center; color: #666;'>
            AlgoTrader Pro v2.0 | Exchange: {current_exchange.title()} | 
            Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()