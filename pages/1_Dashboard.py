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
import bcrypt

# Initialize logger
logger = get_trading_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Dashboard - AlgoTraderPro V2.0",
    page_icon="📈",
    layout="wide"
)

# Header
st.markdown("""
<div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0;'>📈 Trading Dashboard</h1>
    <p style='color: white; margin: 0.5rem 0 0 0;'>Monitor and manage your trading activities for {}</p>
</div>
""".format(st.session_state.get('user', {}).get('username', 'N/A')), unsafe_allow_html=True)

def to_float(value):
    """Safely convert value to float"""
    try:
        return float(value or 0)
    except Exception:
        return 0.0

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user against database with hashed password"""
    try:
        with db_manager.get_session() as session:
            from db import User
            user = session.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                st.session_state.user = user.to_dict()
                st.session_state.authenticated = True
                logger.info(f"User {username} authenticated successfully")
                return True
            return False
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False

def register_user(username: str, password: str) -> bool:
    """Register new user with hashed password"""
    try:
        with db_manager.get_session() as session:
            from db import User
            existing_user = session.query(User).filter_by(username=username).first()
            if existing_user:
                logger.warning(f"User {username} already exists")
                return False
            
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            user = User(
                username=username,
                password_hash=hashed_password,
                binance_api_key=None,
                binance_api_secret=None,
                bybit_api_key=None,
                bybit_api_secret=None
            )
            session.add(user)
            session.commit()
            
            from db import WalletBalance
            session.add(WalletBalance(
                user_id=user.id, 
                account_type="virtual", 
                available=1000.0, 
                used=0.0, 
                total=1000.0,
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
        
        settings = load_settings()
        exchange = settings.get("EXCHANGE", "binance").lower()
        
        trading_engine = TradingEngine()
        trading_engine.switch_exchange(exchange)
        
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
    
    if not st.session_state.get('authenticated', False):
        st.markdown("### 🔐 Dashboard Access")
        st.markdown("""
        **Instructions:**
        - **Login**: Enter your username and password to access your trading dashboard.
        - **Register**: Create a new account if you don't have one. Ensure your password is at least 6 characters.
        - **Note**: Virtual trading is enabled by default with a $1000 starting balance.
        """)
        
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            st.markdown("#### Login to Your Account")
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
            st.markdown("#### Create New Account")
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
    
    try:
        user_data = st.session_state.user
        user_id = user_data['id']
        username = user_data['username']
        current_exchange = st.session_state.get('current_exchange', 'binance')
        account_type = st.session_state.get('account_type', 'virtual')
        
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
            st.markdown(f"""
            <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #10b981;'>
                <h4 style='margin: 0; color: #10b981;'>👤 User</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{username}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if not st.session_state.get('initialized', False):
            with st.spinner("Initializing trading engine..."):
                if not initialize_trading_engine(user_id):
                    st.error("Failed to initialize trading engine. Please refresh the page.")
                    st.stop()
        
        trading_engine = st.session_state.trading_engine
        
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
    
    st.markdown("""
    ### 📉 Real-Time Price Ticker
    **Instructions:**
    - Select cryptocurrency symbols to track real-time prices.
    - Choose an auto-refresh interval to update prices automatically.
    - Monitor key metrics like wallet balance and trade statistics.
    """)
    
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
    
    st.markdown("""
    ### 📊 Key Metrics
    **Instructions:**
    - View your wallet balance, total P&L, win rate, and total trades.
    - Use these metrics to assess your trading performance.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        wallet = db_manager.get_wallet_balance(
            account_type=account_type, 
            user_id=user_id, 
            exchange=current_exchange
        )
        balance = wallet.get('available', 1000.0 if account_type == 'virtual' else 0.0)
        
        stats = trading_engine.get_trade_statistics(account_type) or {}
        
        with col1:
            st.metric("Wallet Balance", format_price(balance))
        with col2:
            st.metric("Total P&L", format_price(stats.get('total_pnl', 0)))
        with col3:
            st.metric("Total Trades", stats.get('total_trades', 0), delta=f"{stats.get('successful_trades', 0)} profitable")
        with col4:
            st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%", delta=f"Recent signals: {len(db_manager.get_signals(limit=10))}")
    
    except Exception as e:
        logger.error(f"Error loading dashboard metrics: {e}")
        st.error(f"Error loading dashboard metrics: {e}")
    
    st.divider()
    
    st.markdown("""
    ### 🔄 Trading Activity
    **Instructions:**
    - **Recent Signals**: View the latest trading signals with details like symbol, side, and entry price.
    - **Open Positions**: Monitor current open trades and their P&L.
    - **Performance Charts**: Analyze cumulative P&L and trade distribution over a selected period.
    """)
    
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
            all_trades = db_manager.get_trades(
                limit=1000,
                virtual=(account_type == 'virtual'), 
                exchange=current_exchange, 
                user_id=user_id
            )
            open_trades = [t for t in all_trades if t.status == 'open']
            
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
            
            all_trades = db_manager.get_trades(
                limit=1000,
                virtual=(account_type == 'virtual'), 
                exchange=current_exchange, 
                user_id=user_id
            )
            all_closed_trades = [t for t in all_trades if t.status == 'closed']
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
    
    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()
    
    trading_status = "🟢 Active" if hasattr(trading_engine, 'is_trading_enabled') and trading_engine.is_trading_enabled() else "🔴 Inactive"
    
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <strong>Status:</strong> Exchange: {current_exchange.title()} | 
        Trading: {trading_status} | 
        Mode: {account_type.title()} | 
        User: {username} | 
        Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()