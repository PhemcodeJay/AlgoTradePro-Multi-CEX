import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime, timezone, timedelta
from db import db_manager, User, WalletBalance
from logging_config import get_trading_logger
from typing import Any
import bcrypt
from settings import validate_env

# Page configuration
st.set_page_config(
    page_title="Trades - AlgoTraderPro V2.0",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Header
st.markdown("""
<div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0;'>💰 Trading Operations</h1>
    <p style='color: white; margin: 0.5rem 0 0 0;'>Manage and execute trades for {}</p>
</div>
""".format(st.session_state.get('user', {}).get('username', 'N/A')), unsafe_allow_html=True)

# Helpers
def is_datetime(value: Any) -> bool:
    return isinstance(value, datetime)

def safe_dt(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return value.strftime(fmt) if is_datetime(value) else "N/A"

def safe_date(value: Any):
    return value.date() if is_datetime(value) else None

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default

def safe_str(value: Any, default: str = "N/A") -> str:
    return str(value) if value is not None else default

def safe_subtract(a: Any, b: Any) -> float:
    return safe_float(a, 0.0) - safe_float(b, 0.0)

def pct(part: float, whole: float) -> float:
    try:
        return (part / whole) * 100 if whole != 0 else 0.0
    except Exception:
        return 0.0

def format_price(value: float) -> str:
    if abs(value) < 1000:
        return f"${value:,.2f}"
    elif abs(value) < 1000000:
        return f"${value / 1000:.2f}K"
    elif abs(value) < 1000000000:
        return f"${value / 1000000:.2f}M"
    else:
        return f"${value / 1000000000:.2f}B"

def highlight_pnl_html(val_str: str) -> str:
    try:
        v = float(str(val_str).replace("$", "").replace(",", ""))
    except Exception:
        v = 0.0
    color = "#2ecc71" if v > 0 else "#ff4d4f" if v < 0 else "#b0b0b0"
    return f'<span style="color:{color};font-weight:600">{val_str}</span>'

# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None

def authenticate_user(username: str, password: str) -> bool:
    try:
        with db_manager.get_session() as session:
            user = session.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                st.session_state.user = {'id': user.id, 'username': user.username}
                st.session_state.authenticated = True
                logger.info(f"User {username} authenticated successfully")
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
                bybit_api_secret=None
            )
            session.add(new_user)
            session.commit()

            session.add(WalletBalance(
                user_id=new_user.id,
                account_type="virtual",
                available=1000.0,
                used=0.0,
                total=1000.0,
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
    st.markdown("### 🔐 Trades Access")
    st.markdown("""
    **Instructions:**
    - **Login**: Enter your username and password to access trade management.
    - **Register**: Create a new account to start trading. Passwords must be at least 6 characters.
    - **Note**: Virtual trading allows you to practice without real funds.
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

# Initialize components
if 'trading_engine' not in st.session_state or st.session_state.trading_engine is None:
    st.error("Trading engine not initialized. Please return to the main page to initialize the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
current_exchange = st.session_state.get('current_exchange', 'binance')
account_type = st.session_state.get('account_type', 'virtual')
automated_trader = st.session_state.get('automated_trader', None)
user_id = st.session_state.user.get('id') if st.session_state.user else None
username = st.session_state.user.get('username', 'N/A')

if not user_id:
    st.error("User not authenticated. Please log in from the main page.")
    st.stop()

# Quick Info Cards
st.markdown("### 📋 Quick Information")
col1, col2, col3, col4 = st.columns(4)
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
with col4:
    wallet = db_manager.get_wallet_balance(account_type, user_id=user_id, exchange=current_exchange)
    balance = safe_float(wallet.get('available'), 1000.0 if account_type == 'virtual' else 0.0)
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #f59e0b;'>
        <h4 style='margin: 0; color: #f59e0b;'>💸 Balance</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{format_price(balance)}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("""
### 📊 Trade Overview
**Instructions:**
- View trade statistics such as total trades, win rate, and P&L.
- Monitor automation status and control automated trading.
- Use tabs to manage open positions, closed trades, manual trading, and performance analytics.
""")

try:
    stats = trading_engine.get_trade_statistics(account_type) or {}
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", stats.get('total_trades', 0))
    with col2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    with col3:
        st.metric("Total P&L", format_price(stats.get('total_pnl', 0)))
except Exception as e:
    logger.error(f"Error loading trade overview: {e}")
    st.error(f"Error loading trade overview: {e}")

st.divider()

st.markdown("""
### 🤖 Automated Trading
**Instructions:**
- Start or stop automated trading to execute signals automatically.
- Reset statistics to clear automation metrics.
- Monitor automation status, last scan time, and executed trades.
""")

col1, col2, col3, col4 = st.columns(4)
trader_status = {}
if automated_trader is not None:
    try:
        trader_status = automated_trader.get_status() or {}
    except Exception as e:
        logger.error(f"Error fetching trader status: {e}")

is_running = bool(trader_status.get('running', False))
is_real_trading_enabled = account_type == 'real' and validate_env(current_exchange, allow_virtual=False)

with col1:
    if is_running:
        st.markdown(f"""
        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #2ecc71;'>
            <h4 style='margin: 0; color: #2ecc71;'>🤖 Status</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Active</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #ff4d4f;'>
            <h4 style='margin: 0; color: #ff4d4f;'>🤖 Status</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>Stopped</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    last_scan = trader_status.get('last_scan_time', 'Never')
    if last_scan and last_scan != 'Never':
        try:
            scan_time = datetime.fromisoformat(last_scan.replace('Z', '+00:00'))
            last_scan = scan_time.strftime('%H:%M:%S')
        except Exception:
            pass
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #636EFA;'>
        <h4 style='margin: 0; color: #636EFA;'>⏰ Last Scan</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{last_scan}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #10b981;'>
        <h4 style='margin: 0; color: #10b981;'>📡 Signals</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{int(trader_status.get('total_signals_generated', 0))}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #f59e0b;'>
        <h4 style='margin: 0; color: #f59e0b;'>💸 Trades</h4>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{int(trader_status.get('successful_trades', 0))}</p>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])
disabled = automated_trader is None
with col1:
    if st.button("Start Automation", type="primary", disabled=disabled, use_container_width=True):
        if automated_trader:
            automated_trader.start()
            st.rerun()

with col2:
    if st.button("Stop Automation", type="primary", disabled=disabled, use_container_width=True):
        if automated_trader:
            automated_trader.stop()
            st.rerun()

with col3:
    if st.button("Reset Statistics", type="primary", disabled=disabled, use_container_width=True):
        if automated_trader:
            automated_trader.reset_statistics()
            st.rerun()

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Open Positions", "Closed Trades", "Manual Trading", "Performance Analytics"])

with tab1:
    st.markdown("""
    ### 📈 Open Positions
    **Instructions:**
    - View all open positions, including unrealized P&L.
    - Refresh the list to update current prices and P&L.
    """)
    try:
        trades = db_manager.get_trades(
            status='open',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )

        if trades:
            trade_data = []
            for t in trades:
                current_price = safe_float(trading_engine.get_current_price(t.symbol, t.exchange), t.entry_price)
                unrealized_pnl = safe_float(t.qty) * (current_price - safe_float(t.entry_price)) if t.side.lower() in ['buy', 'long'] else safe_float(t.qty) * (safe_float(t.entry_price) - current_price)
                
                trade_data.append({
                    "Symbol": safe_str(t.symbol),
                    "Side": safe_str(t.side),
                    "Qty": f"{safe_float(t.qty):.6f}",
                    "Entry": format_price(safe_float(t.entry_price)),
                    "Current": format_price(current_price),
                    "Unrealized P&L": format_price(unrealized_pnl),
                    "Created": safe_dt(t.created_at, "%Y-%m-%d %H:%M")
                })

            df = pd.DataFrame(trade_data)
            st.markdown(f"**Open Positions ({len(trades)})**")
            st.dataframe(df.style.format({"Unrealized P&L": highlight_pnl_html}, escape="html"), use_container_width=True, height=400)
            
            if st.button("🔄 Refresh Open Positions", type="primary", use_container_width=True):
                st.rerun()
        else:
            st.info("📭 No open positions found")

    except Exception as e:
        logger.error(f"Error loading open positions: {e}")
        st.error(f"Error loading open positions: {e}")

with tab2:
    st.markdown("""
    ### ✅ Closed Trades
    **Instructions:**
    - View all closed trades with their final P&L.
    - Analyze P&L distribution and win rate by symbol.
    """)
    try:
        trades = db_manager.get_trades(
            status='closed',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )

        if trades:
            trade_data = []
            for t in trades:
                trade_data.append({
                    "Symbol": safe_str(t.symbol),
                    "Side": safe_str(t.side),
                    "Qty": f"{safe_float(t.qty):.6f}",
                    "Entry": format_price(safe_float(t.entry_price)),
                    "Exit": format_price(safe_float(t.exit_price)),
                    "P&L": format_price(safe_float(t.pnl)),
                    "Closed": safe_dt(t.updated_at, "%Y-%m-%d %H:%M")
                })

            df = pd.DataFrame(trade_data)
            st.markdown(f"**Closed Trades ({len(trades)})**")
            st.dataframe(df.style.format({"P&L": highlight_pnl_html}, escape="html"), use_container_width=True, height=400)

            col1, col2 = st.columns(2)
            with col1:
                pnl_values = [safe_float(t.pnl) for t in trades]
                fig_pnl = px.histogram(x=pnl_values, nbins=20, title="P&L Distribution")
                fig_pnl.update_layout(xaxis_title="P&L ($)", yaxis_title="Number of Trades", height=300)
                st.plotly_chart(fig_pnl, use_container_width=True)

            with col2:
                symbol_perf = {}
                for t in trades:
                    pnl_v = safe_float(getattr(t, "pnl", 0.0))
                    sym = safe_str(getattr(t, "symbol", "N/A"))
                    if sym not in symbol_perf:
                        symbol_perf[sym] = {"wins": 0, "losses": 0}
                    if pnl_v > 0:
                        symbol_perf[sym]["wins"] += 1
                    elif pnl_v < 0:
                        symbol_perf[sym]["losses"] += 1

                if symbol_perf:
                    syms = list(symbol_perf.keys())[:10]
                    winrates = []
                    for s in syms:
                        wins = symbol_perf[s]["wins"]
                        losses = symbol_perf[s]["losses"]
                        total = wins + losses
                        winrates.append((wins / total * 100) if total > 0 else 0)
                    fig_win = px.bar(x=syms, y=winrates, title="Win Rate by Symbol")
                    fig_win.update_layout(xaxis_title="Symbol", yaxis_title="Win Rate (%)", height=300)
                    st.plotly_chart(fig_win, use_container_width=True)

        else:
            st.info("📭 No closed trades found")

    except Exception as e:
        logger.error(f"Error loading closed trades: {e}")
        st.error(f"Error loading closed trades: {e}")

with tab3:
    st.markdown("""
    ### 🎯 Manual Trading
    **Instructions:**
    - Enter trade details to execute a manual trade.
    - Use the trade calculator to assess risk, position size, and potential returns.
    - Real trading requires valid API credentials; virtual trading is always available.
    """)
    if account_type == 'real' and not validate_env(current_exchange, allow_virtual=False):
        st.warning(f"⚠️ Real trading requires {current_exchange.upper()} API credentials. Only virtual trading is available.")
    else:
        st.info(f"{'Virtual' if account_type == 'virtual' else 'Real'} trading mode active on {current_exchange.title()}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Trade Entry**")
        with st.form("trade_form"):
            symbol = st.text_input("Symbol", value="BTCUSDT")
            side = st.selectbox("Side", ["BUY", "SELL", "LONG", "SHORT"])
            entry_price = st.number_input("Entry Price", min_value=0.000001, value=1.0, step=0.000001, format="%.6f")
            
            try:
                calculated_qty = trading_engine.calculate_position_size(symbol, entry_price)
                quantity = st.number_input("Quantity", min_value=0.000001, value=calculated_qty, step=0.000001, format="%.6f")
            except Exception:
                quantity = st.number_input("Quantity", min_value=0.000001, value=0.01, step=0.000001, format="%.6f")

            tp_price = st.number_input("Take Profit", min_value=0.0, value=0.0, step=0.000001, format="%.6f")
            sl_price = st.number_input("Stop Loss", min_value=0.0, value=0.0, step=0.000001, format="%.6f")
            
            disabled = account_type == 'real' and not validate_env(current_exchange, allow_virtual=False)
            submit_trade = st.form_submit_button("📈 Execute Trade", type="primary", disabled=disabled, use_container_width=True)
            
            if submit_trade:
                try:
                    signal = {
                        'symbol': symbol,
                        'side': side,
                        'entry': entry_price,
                        'tp': tp_price if tp_price > 0 else None,
                        'sl': sl_price if sl_price > 0 else None,
                        'leverage': 10,
                        'margin_usdt': 5.0,
                        'trail': 0.0,
                        'market': 'Manual',
                        'indicators': {},
                        'user_id': user_id,
                        'exchange': current_exchange
                    }
                    if account_type == 'virtual':
                        executed = trading_engine.execute_virtual_trade(signal)
                    else:
                        executed = trading_engine.execute_trade(signal)
                    if executed:
                        st.success(f"{'Virtual' if account_type == 'virtual' else 'Real'} trade executed: {side} {quantity} {symbol} at ${entry_price} on {current_exchange.title()}")
                        st.rerun()
                    else:
                        st.error("Failed to execute trade")
                except Exception as e:
                    logger.error(f"Error executing trade: {e}")
                    st.error(f"Error executing trade: {e}")

    with col2:
        st.markdown("**Trade Calculator**")
        try:
            wallet = db_manager.get_wallet_balance(account_type, user_id=user_id, exchange=current_exchange)
            balance = safe_float(wallet.get('available'), 1000.0 if account_type == 'virtual' else 0.0)
            st.metric("Available Balance", format_price(balance))

            risk_percent = st.slider("Risk %", 0.1, 10.0, 1.0, 0.1)
            leverage = st.slider("Leverage", 1, 20, 10)

            risk_amount = balance * (risk_percent / 100)
            position_value = risk_amount * leverage

            if entry_price > 0:
                position_size = position_value / entry_price
                st.write(f"**Risk Amount:** {format_price(risk_amount)}")
                st.write(f"**Position Value:** {format_price(position_value)}")
                st.write(f"**Position Size:** {position_size:.6f}")

                if tp_price > 0 and sl_price > 0:
                    if side.upper() in ["BUY", "LONG"]:
                        potential_profit = (tp_price - entry_price) * position_size
                        potential_loss = (entry_price - sl_price) * position_size
                    else:
                        potential_profit = (entry_price - tp_price) * position_size
                        potential_loss = (sl_price - entry_price) * position_size

                    st.write(f"**Potential Profit:** {format_price(potential_profit)}")
                    st.write(f"**Potential Loss:** {format_price(potential_loss)}")
                    if potential_loss != 0:
                        st.write(f"**R:R Ratio:** 1:{abs(potential_profit/potential_loss):.2f}")

        except Exception as e:
            logger.error(f"Error in trade calculator: {e}")
            st.error(f"Error in trade calculator: {e}")

with tab4:
    st.markdown("""
    ### 📊 Performance Analytics
    **Instructions:**
    - Select a date range to analyze trade performance.
    - View monthly P&L, win rate, and trade duration distribution.
    """)
    try:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=(datetime.now(timezone.utc) - timedelta(days=30)).date(),
                max_value=datetime.now(timezone.utc).date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(timezone.utc).date(),
                min_value=start_date,
                max_value=datetime.now(timezone.utc).date()
            )

        raw_trades = db_manager.get_trades(
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )

        def in_date_range(t):
            ref = getattr(t, "updated_at", None) or getattr(t, "created_at", None)
            return isinstance(ref, datetime) and start_date <= ref.date() <= end_date

        filtered_trades = [t for t in raw_trades if in_date_range(t)]

        if filtered_trades:
            monthly_data = {}
            for t in filtered_trades:
                ref = getattr(t, "updated_at", None) or getattr(t, "created_at", None)
                if not is_datetime(ref):
                    continue
                pnl_v = safe_float(getattr(t, "pnl", 0.0))
                month_key = ref.strftime('%Y-%m') if isinstance(ref, datetime) else "Unknown"
                monthly_data.setdefault(month_key, {'pnl': 0.0, 'trades': 0, 'wins': 0})
                monthly_data[month_key]['pnl'] += pnl_v
                monthly_data[month_key]['trades'] += 1
                if pnl_v > 0:
                    monthly_data[month_key]['wins'] += 1

            months = sorted(monthly_data.keys())
            monthly_pnl = [monthly_data[m]['pnl'] for m in months]
            monthly_winrate = [(monthly_data[m]['wins'] / monthly_data[m]['trades'] * 100) if monthly_data[m]['trades'] > 0 else 0 for m in months]

            col1, col2 = st.columns(2)
            with col1:
                fig_pnl = px.bar(x=months, y=monthly_pnl, title="Monthly P&L")
                fig_pnl.update_layout(xaxis_title="Month", yaxis_title="P&L ($)", height=350)
                st.plotly_chart(fig_pnl, use_container_width=True)

            with col2:
                fig_winrate = px.line(x=months, y=monthly_winrate, title="Monthly Win Rate")
                fig_winrate.update_layout(xaxis_title="Month", yaxis_title="Win Rate (%)", height=350)
                st.plotly_chart(fig_winrate, use_container_width=True)

            st.markdown("**Trade Duration Analysis**")
            durations = []
            for t in filtered_trades:
                created = getattr(t, "created_at", None)
                updated = getattr(t, "updated_at", None)
                if is_datetime(created) and is_datetime(updated):
                    try:
                        dur_hours = (updated - created).total_seconds() / 3600
                        durations.append(dur_hours)
                    except Exception:
                        continue

            if durations:
                fig_duration = px.histogram(x=durations, nbins=20, title="Trade Duration Distribution")
                fig_duration.update_layout(xaxis_title="Duration (Hours)", yaxis_title="Count", height=350)
                st.plotly_chart(fig_duration, use_container_width=True)

                st.write(f"**Average Trade Duration:** {sum(durations)/len(durations):.1f} hours")
                st.write(f"**Shortest Trade:** {min(durations):.1f} hours")
                st.write(f"**Longest Trade:** {max(durations):.1f} hours")
        else:
            st.info("📭 No trades found for selected period")

    except Exception as e:
        logger.error(f"Error loading performance analytics: {e}")
        st.error(f"Error loading performance analytics: {e}")

st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <strong>Status:</strong> Exchange: {current_exchange.title()} | 
    Mode: {account_type.title()} | 
    User: {username} | 
    Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass