# pages/3_Trades.py
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

# Global theme (hybrid): small CSS injection to style buttons, headings, tables, and mimic Binance feel
GLOBAL_CSS = """
<style>
/* Background and card feel - WHITE THEME */
body { background-color: #ffffff; color: #1a1a1a; }
/* Streamlit main content area */
.stApp { background-color: #ffffff; }
/* Headings */
h1, h2, h3, h4, h5 { color: #a8b7e7ff !important; }
/* Buttons */
.stButton>button { background-color: #a8b7e7ff; color: #ffffff; font-weight: 600; border-radius: 6px; }
.stButton>button:hover { filter: brightness(0.95); }
/* Metric value colour (override) */
[data-testid="metric-container"] div[role="heading"] { color: #1a1a1a !important; }
/* Tables created via markdown (we style our own HTML) */
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Helpers — defensive conversions so we never rely on SQLAlchemy truthiness
def is_datetime(value: Any) -> bool:
    return isinstance(value, datetime)

def safe_dt(value: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return formatted datetime or 'N/A'."""
    return value.strftime(fmt) if is_datetime(value) else "N/A"

def safe_date(value: Any):
    """Return date object or None."""
    return value.date() if is_datetime(value) else None

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float; treat None as default (0.0 by default)."""
    try:
        return float(value) if value is not None else default
    except Exception:
        return default

def safe_str(value: Any, default: str = "N/A") -> str:
    return str(value) if value is not None else default

def safe_subtract(a: Any, b: Any) -> float:
    """Subtract using safe_float; None treated as 0.0."""
    return safe_float(a, 0.0) - safe_float(b, 0.0)

def pct(part: float, whole: float) -> float:
    try:
        return (part / whole) * 100 if whole != 0 else 0.0
    except Exception:
        return 0.0

def format_price(value: float) -> str:
    """Formats a float value into a string with K, M, B suffixes."""
    if abs(value) < 1000:
        return f"${value:,.2f}"
    elif abs(value) < 1000000:
        return f"${value / 1000:.2f}K"
    elif abs(value) < 1000000000:
        return f"${value / 1000000:.2f}M"
    else:
        return f"${value / 1000000000:.2f}B"

# Styling helpers (Binance-style: dark-ish header, yellow accents)
BANNER_HTML = """
<div style="background-color:#f8f9fa;padding:12px;border-radius:8px;margin-bottom:8px;border:1px solid #dee2e6;">
  <h2 style="color:#a8b7e7ff;margin:0 0 4px 0">💰 Trading Operations</h2>
  <div style="color:#6c757d;font-size:14px">Interactive trades dashboard — virtual & real modes</div>
</div>
"""
def highlight_pnl_html(val_str: str) -> str:
    """Return HTML span with color depending on sign (expects $ prefix or numeric)."""
    try:
        v = float(str(val_str).replace("$", "").replace(",", ""))
    except Exception:
        v = 0.0
    color = "#2ecc71" if v > 0 else "#ff4d4f" if v < 0 else "#b0b0b0"
    return f'<span style="color:{color};font-weight:600">{val_str}</span>'

# Initialize components & logger
logger = get_trading_logger(__name__)

# Authentication
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

# Initialize required session_state objects
if 'trading_engine' not in st.session_state or st.session_state.trading_engine is None:
    st.error("Trading engine not initialized. Please return to the main page to initialize the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
current_exchange = st.session_state.get('current_exchange', 'binance')
account_type = st.session_state.get('account_type', 'virtual')
automated_trader = st.session_state.get('automated_trader')
user_id = st.session_state.user.get('id') if st.session_state.user else None

if not user_id:
    st.error("User not authenticated. Please log in from the main page.")
    st.stop()

# Render header banner (Binance-style)
st.markdown(BANNER_HTML, unsafe_allow_html=True)

# Trade statistics overview
st.subheader("📊 Trade Overview")

try:
    stats = trading_engine.get_trade_statistics(account_type) or {}
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"{account_type.title()} Trades", stats.get('total_trades', 0))

    with col2:
        st.metric(f"Total {account_type.title()} P&L", format_price(safe_float(stats.get('total_pnl', 0))))

    with col3:
        st.metric(f"{account_type.title()} Win Rate", f"{safe_float(stats.get('win_rate', 0)):.1f}%")

    with col4:
        st.metric(f"Avg {account_type.title()} P&L", format_price(safe_float(stats.get('avg_pnl', 0))))

except Exception as e:
    logger.error(f"Error loading trade statistics: {e}")
    st.error(f"Error loading trade statistics: {e}")

st.divider()

# Automation controls
st.subheader("🤖 Automated Trading")
col1, col2, col3, col4 = st.columns(4)

if automated_trader:
    try:
        trader_status = automated_trader.get_status() or {}
    except Exception as e:
        logger.error(f"Error fetching trader status: {e}")
        trader_status = {}

    is_running = bool(trader_status.get('running', False))
    is_real_trading_enabled = account_type == 'real' and validate_env(current_exchange, allow_virtual=False)

    with col1:
        if is_running:
            st.success("🟢 Automation Active")
        else:
            st.error("🔴 Automation Stopped")

    with col2:
        last_scan = trader_status.get('last_scan_time', 'Never')
        if last_scan and last_scan != 'Never':
            try:
                scan_time = datetime.fromisoformat(last_scan)
                last_scan = scan_time.strftime('%H:%M:%S')
            except Exception:
                pass
        st.metric("Last Scan", last_scan)

    with col3:
        st.metric("Signals Generated", int(trader_status.get('total_signals_generated', 0)))

    with col4:
        st.metric("Trades Executed", int(trader_status.get('successful_trades', 0)))
else:
    st.warning("Automated trader not initialized")

# Control buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if automated_trader:
        running = bool(trader_status.get('running', False))
        if running:
            if st.button("⏹️ Stop Automation", type="primary"):
                try:
                    if automated_trader.stop():
                        st.success("Automated trading stopped")
                        st.rerun()
                    else:
                        st.error("Failed to stop automation")
                except Exception as e:
                    logger.error(f"Stop automation failed: {e}")
                    st.error(f"Failed to stop automation: {e}")
        else:
            disabled = account_type == 'real' and not is_real_trading_enabled
            if st.button("▶️ Start Automation", type="primary", disabled=disabled):
                try:
                    if account_type == 'real' and not is_real_trading_enabled:
                        st.error(f"Real trading requires {current_exchange.upper()} API credentials")
                    elif automated_trader.start():
                        st.success(f"Automated {account_type.title()} trading started")
                        st.rerun()
                    else:
                        st.error("Failed to start automation")
                except Exception as e:
                    logger.error(f"Start automation failed: {e}")
                    st.error(f"Failed to start automation: {e}")

with col2:
    if automated_trader:
        disabled = account_type == 'real' and not is_real_trading_enabled
        if st.button("⚡ Force Scan", disabled=disabled):
            try:
                if account_type == 'real' and not is_real_trading_enabled:
                    st.error(f"Real trading requires {current_exchange.upper()} API credentials")
                else:
                    result = automated_trader.force_scan()
                    if isinstance(result, dict) and result.get('success'):
                        st.success(result.get('message', 'Scan completed'))
                        st.rerun()
                    else:
                        st.error(result.get('error', 'Scan failed') if isinstance(result, dict) else 'Scan failed')
            except Exception as e:
                logger.error(f"Force scan error: {e}")
                st.error(f"Force scan failed: {e}")

with col3:
    with st.expander("⚙️ Automation Settings"):
        col1, col2 = st.columns(2)
        with col1:
            scan_interval = st.number_input(
                "Scan Interval (seconds)",
                min_value=60,
                max_value=86400,
                value=int(getattr(automated_trader, "scan_interval", 3600)) if automated_trader else 3600,
                step=60
            )
            top_n_signals = st.number_input(
                "Top N Signals",
                min_value=1,
                max_value=50,
                value=int(getattr(automated_trader, "top_n_signals", 10)) if automated_trader else 10
            )
        with col2:
            auto_trading = st.checkbox(
                "Enable Auto Trading",
                value=bool(getattr(automated_trader, "auto_trading_enabled", False)) if automated_trader else False
            )
            notifications = st.checkbox(
                "Enable Notifications",
                value=bool(getattr(automated_trader, "notification_enabled", True)) if automated_trader else True
            )

        if st.button("💾 Save Settings"):
            if automated_trader:
                settings = {
                    'scan_interval': int(scan_interval),
                    'top_n_signals': int(top_n_signals),
                    'auto_trading_enabled': bool(auto_trading),
                    'notification_enabled': bool(notifications)
                }
                try:
                    if automated_trader.update_settings(settings):
                        st.success("Settings updated")
                        st.rerun()
                    else:
                        st.error("Failed to update settings")
                except Exception as e:
                    logger.error(f"Update settings failed: {e}")
                    st.error(f"Failed to save settings: {e}")

st.divider()

# Trade management tabs
tab1, tab2, tab3, tab4 = st.tabs(["Open Positions", "Closed Trades", "Manual Trading", "Performance Analytics"])

# Open Positions
with tab1:
    st.subheader("🔓 Open Positions")

    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        symbol_filter = st.text_input("Filter by Symbol (e.g., BTCUSDT)", "").upper()
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        trades = db_manager.get_trades(
            status='open',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            symbol=symbol_filter if symbol_filter else None
        )
        if trades:
            data = []
            for t in trades:
                current_price = safe_float(trading_engine.get_current_price(t.symbol), 0.0) if hasattr(t, "symbol") else 0.0
                entry_price = safe_float(getattr(t, "entry_price", 0.0))
                qty = safe_float(getattr(t, "qty", 0.0))
                side = safe_str(getattr(t, "side", "N/A")).upper()
                if side in ["BUY", "LONG"]:
                    unrealized_pnl = (current_price - entry_price) * qty
                else:
                    unrealized_pnl = (entry_price - current_price) * qty
                data.append({
                    "Symbol": safe_str(getattr(t, "symbol", "N/A")),
                    "Side": side,
                    "Quantity": f"{qty:.6f}",
                    "Entry Price": format_price(entry_price),
                    "Current Price": format_price(current_price),
                    "Unrealized P&L": format_price(unrealized_pnl),
                    "Opened": safe_dt(getattr(t, "created_at", None))
                })
            df_open = pd.DataFrame(data)
            st.dataframe(df_open, use_container_width=True, height=300)
        else:
            st.info("No open positions found")
    except Exception as e:
        logger.error(f"Error loading open positions: {e}")
        st.error(f"Error loading open positions: {e}")

# Closed Trades
with tab2:
    st.subheader("🔒 Closed Trades")

    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        symbol_filter_closed = st.text_input("Filter by Symbol (e.g., BTCUSDT)", key="closed_symbol_filter").upper()
    with col_refresh:
        if st.button("🔄 Refresh", key="closed_refresh"):
            st.rerun()

    try:
        trades = db_manager.get_trades(
            status='closed',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            symbol=symbol_filter_closed if symbol_filter_closed else None,
            limit=2000
        )
        if trades:
            data = []
            for t in trades:
                data.append({
                    "Symbol": safe_str(getattr(t, "symbol", "N/A")),
                    "Side": safe_str(getattr(t, "side", "N/A")).upper(),
                    "Quantity": f"{safe_float(getattr(t, 'qty', 0.0)):.6f}",
                    "Entry Price": format_price(safe_float(getattr(t, "entry_price", 0.0))),
                    "Exit Price": format_price(safe_float(getattr(t, "exit_price", 0.0))),
                    "P&L": format_price(safe_float(getattr(t, "pnl", 0.0))),
                    "Closed": safe_dt(getattr(t, "updated_at", None) or getattr(t, "created_at", None))
                })
            df_closed = pd.DataFrame(data)

            def html_for_closed(df: pd.DataFrame) -> str:
                headers = "".join(f"<th style='background:#f8f9fa;padding:8px;border-bottom:2px solid #dee2e6;color:#1a1a1a'>{col}</th>" for col in df.columns)
                rows_html = []
                for _, row in df.iterrows():
                    cells = [
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6'>{row['Symbol']}</td>",
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6;color:{'#2ecc71' if row['Side'] in ['BUY', 'LONG'] else '#ff4d4f'}'>{row['Side']}</td>",
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6'>{row['Quantity']}</td>",
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6'>{row['Entry Price']}</td>",
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6'>{row['Exit Price']}</td>",
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6'>{highlight_pnl_html(row['P&L'])}</td>",
                        f"<td style='padding:8px;border-bottom:1px solid #dee2e6'>{row['Closed']}</td>",
                    ]
                    rows_html.append(f"<tr>{''.join(cells)}</tr>")
                table = f"<table style='width:100%;border-collapse:collapse;margin-top:8px;background:#ffffff'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
                return table

            st.markdown(html_for_closed(df_closed), unsafe_allow_html=True)

            # Performance mini-charts
            st.subheader("📈 Performance Analysis")
            col1, col2 = st.columns(2)
            with col1:
                pnl_values = [safe_float(getattr(t, "pnl", 0.0)) for t in trades]
                if pnl_values:
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
            st.info("No closed trades found")

    except Exception as e:
        logger.error(f"Error loading closed trades: {e}")
        st.error(f"Error loading closed trades: {e}")

# Manual Trading
with tab3:
    st.subheader("🎯 Manual Trading")
    if account_type == 'real' and not validate_env(current_exchange, allow_virtual=False):
        st.warning(f"⚠️ Real trading requires {current_exchange.upper()} API credentials. Only virtual trading is available.")
    else:
        st.info(f"{'Virtual' if account_type == 'virtual' else 'Real'} trading mode active on {current_exchange.title()}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Trade Entry:**")
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
        if st.button("📈 Execute Trade", type="primary", disabled=disabled):
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
        st.write("**Trade Calculator:**")
        try:
            wallet = db_manager.get_wallet_balance(account_type, user_id=user_id, exchange=current_exchange)
            balance = safe_float(wallet.get('available'), 100.0 if account_type == 'virtual' else 0.0)
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

# Performance Analytics
with tab4:
    st.subheader("📊 Performance Analytics")
    try:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=(datetime.now(timezone.utc) - timedelta(days=30)).date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(timezone.utc).date(),
                min_value=start_date
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

            # Trade duration analysis
            st.write("**Trade Duration Analysis:**")
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
            st.info("No trades found for selected period")

    except Exception as e:
        logger.error(f"Error loading performance analytics: {e}")
        st.error(f"Error loading performance analytics: {e}")

# Status footer
try:
    open_count = len(db_manager.get_trades(
        status='open',
        virtual=(account_type == 'virtual'),
        exchange=current_exchange,
        user_id=user_id,
        limit=2000
    ))
except Exception:
    open_count = 0

st.markdown(f"""
---
**Status:** Exchange: {safe_str(current_exchange).title()} |
Mode: {safe_str(account_type).title()} |
Open Positions: {open_count} |
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""", unsafe_allow_html=True)