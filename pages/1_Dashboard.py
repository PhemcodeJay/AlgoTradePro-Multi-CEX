import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
from db import db_manager
from logging_config import get_trading_logger
from utils import format_price, format_number
import os

# Initialize logger
logger = get_trading_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Dashboard - AlgoTraderPro V2.0",
    page_icon="📈",
    layout="wide"
)

def to_float(value):
    """Safely convert value to float"""
    try:
        return float(value or 0)
    except Exception:
        return 0.0

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user against database"""
    try:
        with db_manager.get_session() as session:
            from db import User
            user = session.query(User).filter_by(username=username).first()
            if user and user.password_hash == password:  # Simple check - use proper hashing in production
                st.session_state.user = user.to_dict()
                st.session_state.authenticated = True
                logger.info(f"User {username} authenticated successfully")
                return True
            return False
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False

def register_user(username: str, password: str) -> bool:
    """Register new user"""
    try:
        with db_manager.get_session() as session:
            from db import User
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                logger.warning(f"User {username} already exists")
                return False
            
            user = User(
                username=username,
                password_hash=password,  # In production, hash this properly
                binance_api_key=None,
                binance_api_secret=None,
                bybit_api_key=None,
                bybit_api_secret=None
            )
            session.add(user)
            session.commit()
            
            # Initialize wallet balances for new user
            from db import WalletBalance
            session.add(WalletBalance(
                user_id=user.id, 
                account_type="virtual", 
                available=10000.0, 
                used=0.0, 
                total=10000.0,
                currency="USDT"
            ))
            session.add(WalletBalance(
                user_id=user.id, 
                account_type="real", 
                available=0.0, 
                used=0.0, 
                total=0.0,
                currency="USDT"
            ))
            session.commit()
            
            logger.info(f"User {username} registered successfully")
            return True
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return False

def initialize_trading_engine(user_id: int):
    """Initialize trading engine for authenticated user"""
    try:
        from multi_trading_engine import TradingEngine
        from settings import load_settings
        
        # Load settings
        settings = load_settings()
        exchange = settings.get("EXCHANGE", "binance").lower()
        
        # Initialize trading engine
        trading_engine = TradingEngine()
        trading_engine.switch_exchange(exchange)
        
        # Store in session state
        st.session_state.trading_engine = trading_engine
        st.session_state.current_exchange = exchange
        st.session_state.user_id = user_id
        st.session_state.initialized = True
        
        logger.info(f"Trading engine initialized for user {user_id}, exchange: {exchange}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize trading engine: {e}")
        st.error(f"Failed to initialize trading engine: {str(e)}")
        return False

def main():
    """Main dashboard function"""
    
    # Check authentication
    if not st.session_state.get('authenticated', False):
        st.title("🔐 Dashboard Access")
        st.markdown("**Please login or register to access the trading dashboard**")
        
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            st.markdown("### Login to your account")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Login", use_container_width=True)
                
                if submit_login:
                    if username and password:
                        with st.spinner("Authenticating..."):
                            if authenticate_user(username, password):
                                st.success(f"✅ Welcome back, {username}!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid username or password")
                    else:
                        st.error("Please enter both username and password")
        
        with register_tab:
            st.markdown("### Create new account")
            with st.form("register_form"):
                new_username = st.text_input("New Username", placeholder="Choose a username")
                new_password = st.text_input("New Password", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                submit_register = st.form_submit_button("Register", use_container_width=True)
                
                if submit_register:
                    if not all([new_username, new_password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        with st.spinner("Creating account..."):
                            if register_user(new_username, new_password):
                                st.success("✅ Registration successful! Please login with your credentials.")
                            else:
                                st.error("❌ Registration failed. Username may already exist.")
        
        st.stop()
    
    # User is authenticated, initialize components
    try:
        user_data = st.session_state.user
        user_id = user_data['id']
        username = user_data['username']
        
        # Welcome message
        st.title(f"📈 Trading Dashboard")
        st.markdown(f"👋 Welcome, **{username}**!")
        
        # Initialize trading engine if not done
        if not st.session_state.get('initialized', False):
            with st.spinner("Initializing trading engine..."):
                if not initialize_trading_engine(user_id):
                    st.error("Failed to initialize trading engine. Please refresh the page.")
                    st.stop()
        
        trading_engine = st.session_state.trading_engine
        current_exchange = st.session_state.get('current_exchange', 'binance')
        account_type = st.session_state.get('account_type', 'virtual')
        
        # Logout button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col3:
            if st.button("🚪 Logout", type="secondary", use_container_width=False):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Logged out successfully!")
                st.rerun()
        
        st.divider()
        
    except Exception as e:
        logger.error(f"Error in dashboard initialization: {e}")
        st.error(f"Dashboard initialization error: {str(e)}")
        st.stop()
    
    # Real-Time Price Ticker
    st.subheader("📉 Real-Time Price Ticker")
    
    default_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    all_symbols = default_symbols + ["XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT"]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_symbols = st.multiselect(
            "Select symbols to track",
            all_symbols,
            default=default_symbols,
            key="ticker_symbols"
        )
    
    with col2:
        refresh_interval = st.selectbox(
            "Auto-refresh interval",
            [0, 5, 10, 30, 60],
            index=0,
            format_func=lambda x: "Off" if x == 0 else f"Every {x} seconds",
            key="ticker_refresh"
        )
    
    if selected_symbols:
        price_placeholder = st.empty()
        prices = {}
        with price_placeholder.container():
            price_cols = st.columns(len(selected_symbols))
            for i, sym in enumerate(selected_symbols):
                try:
                    price = trading_engine.client.get_current_price(sym)
                    prices[sym] = to_float(price)
                except Exception as e:
                    prices[sym] = None
                    logger.warning(f"Failed to fetch price for {sym}: {str(e)}")
                
                with price_cols[i]:
                    value = format_price(prices[sym]) if prices[sym] is not None else "N/A"
                    st.metric(sym, value)
    else:
        st.info("⚠️ Select symbols to display real-time prices")
    
    st.divider()
    
    # Main metrics row
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get wallet balance with user_id and exchange
        wallet = db_manager.get_wallet_balance(
            balance_type=account_type, 
            user_id=user_id, 
            exchange=current_exchange
        )
        balance = wallet.get('available', 10000.0 if account_type == 'virtual' else 0.0)
        
        # Get trade statistics
        stats = trading_engine.get_trade_statistics(account_type)
        
        # Get recent signals and open trades
        recent_signals = db_manager.get_signals(limit=24, exchange=current_exchange, user_id=user_id)
        open_trades = db_manager.get_trades(
            status='open', 
            virtual=(account_type == 'virtual'), 
            exchange=current_exchange, 
            user_id=user_id
        )
        
        with col1:
            st.metric(
                f"{account_type.title()} Balance",
                format_price(balance),
                delta=format_price(stats.get('total_pnl', 0)),
                help="Total account balance including unrealized P&L"
            )
        
        with col2:
            st.metric(
                "Open Positions",
                len(open_trades),
                delta=f"Max: {trading_engine.get_max_open_positions()}",
                help="Number of currently open trades"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                stats.get('total_trades', 0),
                delta=f"{stats.get('successful_trades', 0)} profitable",
                help="Total number of executed trades"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{stats.get('win_rate', 0):.1f}%",
                delta=f"Recent signals: {len(recent_signals)}",
                help="Percentage of profitable trades"
            )
    
    except Exception as e:
        logger.error(f"Error loading dashboard metrics: {e}")
        st.error(f"Error loading dashboard metrics: {e}")
    
    st.divider()
    
    # Trading Activity Overview
    st.subheader("🔄 Trading Activity")
    
    tab1, tab2, tab3 = st.tabs(["Recent Signals", "Open Positions", "Performance Charts"])
    
    with tab1:
        st.write("**Latest Trading Signals**")
        try:
            signals = db_manager.get_signals(limit=10, exchange=current_exchange, user_id=user_id)
            
            if signals:
                signal_data = []
                for signal in signals:
                    created_at = datetime.fromisoformat(signal['created_at']) if signal.get('created_at') else None
                    signal_data.append({
                        'Symbol': signal.get('symbol', 'N/A'),
                        'Side': signal.get('side', 'N/A'),
                        'Score': f"{signal.get('score', 0):.1f}",
                        'Entry': format_price(signal.get('entry', 0), 6),
                        'Market': signal.get('market', 'N/A'),
                        'Created': created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'N/A'
                    })
                
                df_signals = pd.DataFrame(signal_data)
                
                def color_side(val):
                    if str(val).upper() in ['BUY', 'LONG']:
                        return 'color: green'
                    elif str(val).upper() in ['SELL', 'SHORT']:
                        return 'color: red'
                    return 'color: gray'
                
                styled_df = df_signals.style.applymap(color_side, subset=['Side'])
                st.dataframe(styled_df, use_container_width=True, height=300)
                
            else:
                st.info("📭 No recent signals found")
                
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            st.error(f"Error loading signals: {e}")
    
    with tab2:
        st.write("**Current Open Positions**")
        try:
            open_trades = db_manager.get_trades(
                status='open', 
                virtual=(account_type == 'virtual'), 
                exchange=current_exchange, 
                user_id=user_id
            )
            
            if open_trades:
                trade_data = []
                for trade in open_trades:
                    current_pnl = trading_engine.calculate_virtual_pnl({
                        "symbol": trade.symbol,
                        "entry_price": trade.entry_price,
                        "qty": trade.qty,
                        "side": trade.side
                    }) if account_type == 'virtual' else (trade.pnl or 0.0)
                    
                    created_at = trade.created_at
                    trade_data.append({
                        'Symbol': trade.symbol,
                        'Side': trade.side,
                        'Quantity': f"{trade.qty:.6f}",
                        'Entry Price': format_price(trade.entry_price, 6),
                        'Current P&L': format_price(current_pnl),
                        'Opened': created_at.strftime('%Y-%m-%d %H:%M') if created_at else 'N/A'
                    })
                
                df_trades = pd.DataFrame(trade_data)
                
                def color_pnl(val):
                    try:
                        val_float = float(str(val).replace('$', '').replace(',', ''))
                        if val_float < 0:
                            return 'color: red'
                        elif val_float > 0:
                            return 'color: green'
                        return 'color: gray'
                    except:
                        return 'color: gray'
                
                styled_df = df_trades.style.applymap(color_pnl, subset=['Current P&L'])
                st.dataframe(styled_df, use_container_width=True, height=300)
                
            else:
                st.info("✅ No open positions")
                
        except Exception as e:
            logger.error(f"Error loading open positions: {e}")
            st.error(f"Error loading open positions: {e}")
    
    with tab3:
        st.write("**Performance Analytics**")
        
        try:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start date",
                    value=(datetime.now(timezone.utc) - timedelta(days=30)).date(),
                    max_value=datetime.now(timezone.utc).date(),
                    key="perf_start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End date",
                    value=datetime.now(timezone.utc).date(),
                    min_value=start_date,
                    max_value=datetime.now(timezone.utc).date(),
                    key="perf_end_date"
                )
            
            all_closed_trades = db_manager.get_trades(
                status='closed', 
                virtual=(account_type == 'virtual'), 
                exchange=current_exchange, 
                user_id=user_id
            )
            closed_trades = [
                t for t in all_closed_trades
                if t.updated_at and start_date <= t.updated_at.date() <= end_date
            ]
            
            if closed_trades:
                trade_dates = []
                cumulative_pnl = []
                daily_pnl = []
                
                running_pnl = 0
                for trade in sorted(closed_trades, key=lambda t: t.updated_at or t.created_at):
                    if trade.pnl is not None:
                        date = trade.updated_at or trade.created_at
                        trade_dates.append(date)
                        running_pnl += trade.pnl
                        cumulative_pnl.append(running_pnl)
                        daily_pnl.append(trade.pnl)
                
                if trade_dates:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cum = go.Figure()
                        fig_cum.add_trace(go.Scatter(
                            x=trade_dates,
                            y=cumulative_pnl,
                            mode='lines+markers',
                            name='Cumulative P&L',
                            line=dict(color='#10B981', width=3),
                            hovertemplate='Date: %{x}<br>P&L: $%{y:.2f}<extra></extra>'
                        ))
                        fig_cum.update_layout(
                            title="💹 Cumulative P&L Over Time",
                            xaxis_title="Date",
                            yaxis_title="P&L ($)",
                            height=400,
                            hovermode="x unified",
                            showlegend=False
                        )
                        st.plotly_chart(fig_cum, use_container_width=True)
                    
                    with col2:
                        fig_dist = px.histogram(
                            x=daily_pnl,
                            nbins=20,
                            title="📊 Trade P&L Distribution",
                            color_discrete_sequence=['#636EFA']
                        )
                        fig_dist.update_layout(
                            xaxis_title="P&L ($)",
                            yaxis_title="Number of Trades",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    
                    st.write("**📈 Key Performance Metrics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🏆 Best Trade", format_price(max(daily_pnl)) if daily_pnl else "$0.00")
                    with col2:
                        st.metric("💥 Worst Trade", format_price(min(daily_pnl)) if daily_pnl else "$0.00")
                    with col3:
                        st.metric("📊 Avg Trade", format_price(sum(daily_pnl)/len(daily_pnl)) if daily_pnl else "$0.00")
                    with col4:
                        profitable_trades = len([p for p in daily_pnl if p > 0])
                        st.metric("✅ Profitable %", f"{(profitable_trades/len(daily_pnl)*100):.1f}%" if daily_pnl else "0.0%")
            else:
                st.info("📭 No closed trades available for the selected period")
                
        except Exception as e:
            logger.error(f"Error loading performance charts: {e}")
            st.error(f"Error loading performance charts: {e}")
    
    st.divider()
    
    # Auto-refresh logic
    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Status footer
    trading_status = "🟢 Active" if hasattr(trading_engine, 'is_trading_enabled') and trading_engine.is_trading_enabled() else "🔴 Inactive"
    
    st.markdown(f"""
    ---
    **Status:** Exchange: {current_exchange.title()} | 
    Trading: {trading_status} | 
    Mode: {account_type.title()} | 
    User: {username} | 
    Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
    """)

if __name__ == "__main__":
    main()