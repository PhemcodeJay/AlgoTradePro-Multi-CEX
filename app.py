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
    st.session_state.account_type = 'virtual'
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.has_api_keys = False

def to_float(value):
    try:
        return float(value or 0)
    except Exception:
        return 0.0

def authenticate_user(username: str, password: str) -> bool:
    try:
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                st.session_state.user = {'id': user.id, 'username': user.username}
                st.session_state.authenticated = True
                return True
            return False
    except Exception as e:
        logger.error(f"Authentication error for {username}: {e}")
        return False

def register_user(username: str, password: str) -> bool:
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
                bybit_api_secret=None,
                default_exchange="bybit",
                trading_mode="virtual"
            )
            session.add(new_user)
            session.commit()

            for acc_type in ["virtual", "real"]:
                session.add(WalletBalance(
                    user_id=new_user.id,
                    account_type=acc_type,
                    available=100.0 if acc_type == "virtual" else 0.0,
                    used=0.0,
                    total=100.0 if acc_type == "virtual" else 0.0,
                    currency="USDT",
                    exchange="bybit"
                ))
            session.commit()
            logger.info(f"Registered new user: {username}, ID: {new_user.id}")
            return True
    except Exception as e:
        logger.error(f"Registration error for {username}: {e}")
        session.rollback()
        return False

def initialize_app():
    try:
        user_id = st.session_state.user.get("id")
        if not user_id:
            logger.error("No user_id in session")
            st.error("User not authenticated. Please log in.")
            return False

        settings = load_settings(user_id=user_id)
        if not settings:
            st.error("Failed to load user settings.")
            return False

        exchange = settings.get("EXCHANGE", "bybit").lower()
        requested_mode = settings.get("TRADING_MODE", "virtual").lower()

        has_api_keys, _ = validate_env(exchange=exchange, user_id=user_id)
        st.session_state.has_api_keys = has_api_keys

        if not has_api_keys:
            st.warning(f"No {exchange.title()} API credentials found. Virtual Mode Only")
            with st.expander("About Virtual Mode", expanded=False):
                st.markdown(f"""
                ### Virtual Trading Mode
                - Uses real market data
                - Simulates trades
                - Cannot execute real orders
                ---
                **To Enable Real Trading:**
                1. Go to **Settings → Exchange**
                2. Add your **API Key + Secret**
                3. Save and refresh
                """)
            st.session_state.account_type = "virtual"
        else:
            st.session_state.account_type = requested_mode

        try:
            trading_engine = TradingEngine(
                user_id=user_id,
                exchange=exchange,
                account_type=st.session_state.account_type
            )
        except Exception as e:
            logger.error(f"Failed to create TradingEngine: {e}", exc_info=True)
            st.error(f"Trading engine failed to start: {e}")
            return False

        try:
            trading_engine.switch_exchange(exchange)
        except Exception as e:
            logger.error(f"Failed to switch to {exchange}: {e}")
            st.error(f"Could not switch to {exchange.title()}: {e}")
            return False

        if st.session_state.account_type == "real" and has_api_keys:
            try:
                if not trading_engine.enable_trading():
                    st.warning("Real trading enabled but failed to start.")
            except Exception as e:
                logger.error(f"Error enabling real trading: {e}")
                st.warning(f"Real trading could not be enabled: {e}")

        try:
            automated_trader = AutomatedTrader(trading_engine)
        except Exception as e:
            logger.error(f"Failed to create AutomatedTrader: {e}")
            st.error(f"Automated trader failed: {e}")
            return False

        st.session_state.trading_engine = trading_engine
        st.session_state.automated_trader = automated_trader
        st.session_state.current_exchange = exchange
        st.session_state.initialized = True

        logger.info("App initialized", extra={
            "user_id": user_id,
            "exchange": exchange,
            "mode": st.session_state.account_type,
            "has_api_keys": has_api_keys,
        })
        return True

    except Exception as e:
        logger.error(f"Unexpected error in initialize_app: {e}", exc_info=True)
        st.error(f"Initialization failed: {e}")
        return False

@st.cache_data
def get_cached_dashboard_data(exchange: str, account_type: str, _is_virtual: bool) -> Dict[str, Any]:
    """Fetch and cache dashboard data as plain dictionaries"""
    try:
        user_id = st.session_state.user.get('id') if st.session_state.user else None
        if not user_id:
            logger.error("No user_id found in session state for dashboard data")
            return {
                'signals': [],
                'wallet': {'available': 0.0, 'used': 0.0, 'total': 0.0},
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
            all_trades = db_manager.get_trades(limit=10000, virtual=_is_virtual, exchange=exchange, user_id=user_id) or []
            open_trades = [t for t in all_trades if t.status == 'open']
            closed_trades = [t for t in all_trades if t.status == 'closed']
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
            'wallet': {'available': 0.0, 'used': 0.0, 'total': 0.0},
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
            trades = db_manager.get_trades(limit=limit, virtual=_is_virtual, exchange=exchange, user_id=user_id) or []
            return [t.to_dict() for t in trades]
    except Exception as e:
        logger.error(f"Error fetching cached trades: {e}")
        return []

def main():
    # Header
    if st.session_state.authenticated:
        current_exchange = st.session_state.current_exchange
        account_type = st.session_state.account_type
        st.markdown(f"""
        <div style='padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;'>
            <h2 style='color: white; margin: 0;'>📈 Dashboard - {current_exchange.title()} ({account_type.title()})</h2>
        </div>
        """, unsafe_allow_html=True)

    if not st.session_state.authenticated:
        # Authentication form
        st.title("🔐 Login to AlgoTraderPro")
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab_register:
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Register"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif register_user(new_username, new_password):
                    st.success("Registered successfully! Please login.")
                else:
                    st.error("Registration failed. Username may already exist.")
    else:
        if not st.session_state.initialized:
            if not initialize_app():
                st.stop()

        # Sidebar controls
        st.sidebar.header("⚙️ Controls")

        # Exchange selection
        exchange_options = ["Binance", "Bybit"]
        current_exchange_index = 0 if st.session_state.current_exchange == "binance" else 1
        selected_exchange = st.sidebar.selectbox(
            "Select Exchange",
            exchange_options,
            index=current_exchange_index,
            key="exchange_select"
        ).lower()

        # Handle exchange switch
        if selected_exchange != st.session_state.current_exchange:
            try:
                st.session_state.current_exchange = selected_exchange
                if st.session_state.trading_engine:
                    st.session_state.trading_engine.switch_exchange(selected_exchange)
                    # Update settings with new exchange
                    user_id = st.session_state.user.get('id')
                    if user_id:
                        settings = load_settings(user_id=user_id)
                        settings["EXCHANGE"] = selected_exchange
                        save_settings(settings, user_id=user_id)
                    # Clear cache and rerun
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                logger.error(f"Error switching exchange to {selected_exchange}: {e}")
                st.error(f"Failed to switch exchange: {e}")

        # Account selection with disable if no API keys
        options = ["Virtual", "Real"] if st.session_state.has_api_keys else ["Virtual"]
        index = 0 if st.session_state.account_type == 'virtual' else 1
        account_type = st.sidebar.selectbox(
            "Account Type",
            options,
            index=min(index, len(options)-1),
            key="account_type_select"
        ).lower()

        if account_type != st.session_state.account_type:
            st.session_state.account_type = account_type
            if st.session_state.trading_engine:
                st.session_state.trading_engine.set_account_type(account_type)
                # Update settings with new trading mode
                user_id = st.session_state.user.get('id')
                if user_id:
                    settings = load_settings(user_id=user_id)
                    settings["TRADING_MODE"] = account_type
                    save_settings(settings, user_id=user_id)
            st.cache_data.clear()
            st.rerun()

        current_exchange = st.session_state.current_exchange

        # Mode info
        if account_type == 'virtual':
            st.sidebar.info("🧪 Virtual mode uses simulated balances")
        else:
            st.sidebar.warning("⚠️ Real mode uses live market data and executes actual trades")

        st.sidebar.divider()

        # Trading controls
        st.sidebar.subheader("🎯 Trading Status")

        if st.session_state.trading_engine:
            trading_enabled = bool(st.session_state.trading_engine.is_trading_enabled())

            col1, col2 = st.sidebar.columns(2)

            with col1:
                start_disabled = trading_enabled or (account_type == 'real' and not st.session_state.has_api_keys)
                if st.sidebar.button("▶️ Start Trading", disabled=start_disabled, key="start_trading"):
                    try:
                        if st.session_state.trading_engine and st.session_state.trading_engine.enable_trading():
                            st.success("Trading enabled")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Error enabling trading: {e}")
                        st.error("Failed to enable trading")

            with col2:
                if st.sidebar.button("⏸️ Stop Trading", disabled=not trading_enabled, key="stop_trading"):
                    try:
                        if st.session_state.trading_engine and st.session_state.trading_engine.disable_trading("Manual stop"):
                            st.warning("Trading disabled")
                            st.rerun()
                    except Exception as e:
                        logger.error(f"Error disabling trading: {e}")
                        st.error("Failed to disable trading")

            # Emergency stop
            if st.sidebar.button("🚨 Emergency Stop", type="secondary", key="emergency_stop"):
                try:
                    if st.session_state.trading_engine and st.session_state.trading_engine.emergency_stop("Manual emergency stop"):
                        st.error("Emergency stop activated")
                        st.rerun()
                except Exception as e:
                    logger.error(f"Error during emergency stop: {e}")
                    st.error("Emergency stop failed")

            # Status indicators
            if trading_enabled:
                st.sidebar.success("🟢 Trading Active")
            else:
                st.sidebar.error("🔴 Trading Inactive")

        st.sidebar.divider()

        # Quick stats
        st.sidebar.subheader("📊 Quick Stats")

        if st.session_state.trading_engine:
            try:
                user_id = st.session_state.user.get('id')
                if not user_id:
                    st.error("User ID not found. Please log in again.")
                    st.stop()

                virtual_wallet = db_manager.get_wallet_balance(account_type='virtual', user_id=user_id, exchange=current_exchange)
                real_wallet = db_manager.get_wallet_balance(account_type='real', user_id=user_id, exchange=current_exchange)
                stats_virtual = st.session_state.trading_engine.get_trade_statistics('virtual')
                stats_real = st.session_state.trading_engine.get_trade_statistics('real')

                combined_total_trades = int(stats_virtual.get('total_trades', 0)) + int(stats_real.get('total_trades', 0))
                combined_win_rate = 0.0
                if combined_total_trades > 0:
                    wins_virtual = stats_virtual.get('win_rate', 0) / 100 * stats_virtual.get('total_trades', 0)
                    wins_real = stats_real.get('win_rate', 0) / 100 * stats_real.get('total_trades', 0)
                    combined_win_rate = ((wins_virtual + wins_real) / combined_total_trades) * 100
                combined_pnl = float(stats_virtual.get('total_pnl', 0)) + float(stats_real.get('total_pnl', 0))

                colb1, colb2 = st.sidebar.columns(2)
                with colb1:
                    virtual_av = to_float(virtual_wallet.get('available') if virtual_wallet else 0.0)
                    st.metric("Virtual Balance", f"${virtual_av:.2f}")
                with colb2:
                    real_av = to_float(real_wallet.get('available') if real_wallet else 0.0)
                    st.metric("Real Balance", f"${real_av:.2f}")

                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Total Trades", combined_total_trades)
                with col2:
                    st.metric("Win Rate", f"{combined_win_rate:.1f}%")

                st.metric("Total P&L", f"${combined_pnl:.2f}")

            except Exception as e:
                logger.error(f"Error loading quick stats: {e}")
                st.error("Error loading stats")

        # Main content area
        try:
            # Fetch data for both virtual and real
            dashboard_data_virtual = get_cached_dashboard_data(current_exchange, 'virtual', True)
            dashboard_data_real = get_cached_dashboard_data(current_exchange, 'real', False)

            # Virtual Key Metrics
            st.subheader("🧪 Virtual Mode Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            signals = dashboard_data_virtual['signals']
            recent_count = len(signals)
            last_scan = "Never"
            if recent_count > 0 and signals[0].get('created_at') != 'N/A':
                last_scan = signals[0]['created_at'].split(' ')[1][:5]
            with col1:
                st.metric("Recent Signals", recent_count, delta=f"Last scan: {last_scan}")

            open_count_virtual = len(dashboard_data_virtual['open_trades'])
            max_open = getattr(st.session_state.trading_engine, "max_open_positions", "N/A") if st.session_state.trading_engine else "N/A"
            with col2:
                st.metric("Open Positions", open_count_virtual, delta=f"Max: {max_open}")

            with col3:
                balance = dashboard_data_virtual['wallet'].get('available')
                st.metric("Virtual Balance", f"${(balance if balance is not None and balance != '' else 100.0):.2f}")

            today_date = datetime.now(timezone.utc).date().isoformat()
            today_trades_virtual = [t for t in dashboard_data_virtual['closed_trades'] if t['created_at'].startswith(today_date)]
            today_pnl_virtual = sum(t['pnl'] for t in today_trades_virtual)
            pnl_value_virtual = float(to_float(today_pnl_virtual))
            pnl_delta_virtual = "➖"
            if pnl_value_virtual > 0:
                pnl_delta_virtual = "📈"
            elif pnl_value_virtual < 0:
                pnl_delta_virtual = "📉"
            with col4:
                st.metric("Today's P&L", f"${pnl_value_virtual:.2f}", delta=pnl_delta_virtual)

            # Real Key Metrics
            st.subheader("💰 Real Mode Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Recent Signals", recent_count, delta=f"Last scan: {last_scan}")

            open_count_real = len(dashboard_data_real['open_trades'])
            with col2:
                st.metric("Open Positions", open_count_real, delta=f"Max: {max_open}")

            with col3:
                st.metric("Real Balance", f"${dashboard_data_real['wallet']['available']:.2f}")

            today_trades_real = [t for t in dashboard_data_real['closed_trades'] if t['created_at'].startswith(today_date)]
            today_pnl_real = sum(t['pnl'] for t in today_trades_real)
            pnl_value_real = float(to_float(today_pnl_real))
            pnl_delta_real = "➖"
            if pnl_value_real > 0:
                pnl_delta_real = "📈"
            elif pnl_value_real < 0:
                pnl_delta_real = "📉"
            with col4:
                st.metric("Today's P&L", f"${pnl_value_real:.2f}", delta=pnl_delta_real)

        except Exception as e:
            logger.error(f"Error loading dashboard metrics: {e}")
            st.error("Error loading dashboard data")

        # Recent activity
        st.subheader("🔄 Recent Activity")

        tab1, tab2, tab3 = st.tabs(["Latest Signals", "Recent Virtual Trades", "Recent Real Trades"])

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
                recent_trades = get_cached_trades(current_exchange, True, limit=5)
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
                    st.info("No recent virtual trades found")
            except Exception as e:
                logger.error(f"Error loading recent virtual trades: {e}")
                st.error("Error loading trades")

        with tab3:
            try:
                recent_trades = get_cached_trades(current_exchange, False, limit=5)
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
                    st.info("No recent real trades found")
            except Exception as e:
                logger.error(f"Error loading recent real trades: {e}")
                st.error("Error loading trades")

        # Auto-refresh
        if st.button("🔄 Refresh Dashboard", key="refresh_dashboard"):
            st.cache_data.clear()  # Clear cache on refresh
            st.rerun()

        # Footer
        st.divider()
        st.markdown(
            f"""
            <div style='text-align: center; color: #666;'>
                AlgoTrader Pro v2.0 | Exchange: {current_exchange.title()} | Mode: {account_type.title()} |
                Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()