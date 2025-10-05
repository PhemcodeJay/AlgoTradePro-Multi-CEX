# pages/3_Trades.py — fully refactored, interactive (hybrid Binance-style theme)
import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime, timezone, timedelta
from db import db_manager
from logging_config import get_trading_logger
from typing import Any

# Page configuration
st.set_page_config(
    page_title="Trades - AlgoTrader Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Global theme (hybrid): small CSS injection to style buttons, headings, tables, and mimic Binance feel
# -------------------------
GLOBAL_CSS = """
<style>
/* Background and card feel */
body { background-color: #071021; color: #e6eef8; }
/* Streamlit main content area */
.stApp { background-color: #071021; }
/* Headings */
h1, h2, h3, h4, h5 { color: #f3b400 !important; }
/* Buttons */
.stButton>button { background-color: #f3b400; color: #071021; font-weight: 600; border-radius: 6px; }
.stButton>button:hover { filter: brightness(0.95); }
/* Metric value colour (override) */
[data-testid="metric-container"] div[role="heading"] { color: #e6eef8 !important; }
/* Tables created via markdown (we style our own HTML) */
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# -------------------------
# Helpers — defensive conversions so we never rely on SQLAlchemy truthiness
# -------------------------
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

# -------------------------
# Styling helpers (Binance-style: dark-ish header, yellow accents)
# -------------------------
BANNER_HTML = """
<div style="background-color:#0b1220;padding:12px;border-radius:8px;margin-bottom:8px;">
  <h2 style="color:#f3b400;margin:0 0 4px 0">💰 Trading Operations</h2>
  <div style="color:#cbd5e1;font-size:14px">Interactive trades dashboard — virtual & real modes</div>
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

# -------------------------
# Initialize components & logger
# -------------------------
logger = get_trading_logger(__name__)

# Initialize required session_state objects
if 'trading_engine' not in st.session_state or 'current_exchange' not in st.session_state:
    st.error("Trading engine not initialized. Please restart the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
current_exchange = st.session_state.current_exchange
account_type = st.session_state.get('account_type', 'virtual')
automated_trader = st.session_state.get('automated_trader')

# Render header banner (Binance-style)
st.markdown(BANNER_HTML, unsafe_allow_html=True)

# -------------------------
# Trade statistics overview
# -------------------------
st.subheader("📊 Trade Overview")

try:
    stats = trading_engine.get_trade_statistics(account_type) or {}
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"{account_type.title()} Trades", stats.get('total_trades', 0))

    with col2:
        st.metric(f"{account_type.title()} P&L", f"${safe_float(stats.get('total_pnl', 0)):.2f}")

    with col3:
        st.metric(f"{account_type.title()} Win Rate", f"{safe_float(stats.get('win_rate', 0)):.1f}%")

    with col4:
        st.metric(f"Avg {account_type.title()} P&L", f"${safe_float(stats.get('avg_pnl', 0)):.2f}")

except Exception as e:
    logger.error(f"Error loading trade statistics: {e}")
    st.error(f"Error loading trade statistics: {e}")

st.divider()

# -------------------------
# Automation controls
# -------------------------
st.subheader("🤖 Automated Trading")
col1, col2, col3, col4 = st.columns(4)

if automated_trader:
    try:
        trader_status = automated_trader.get_status() or {}
    except Exception:
        trader_status = {}

    is_running = bool(trader_status.get('running', False))

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
                    st.error("Failed to stop automation")
        else:
            if st.button("▶️ Start Automation", type="primary"):
                try:
                    if automated_trader.start():
                        st.success("Automated trading started")
                        st.rerun()
                    else:
                        st.error("Failed to start automation")
                except Exception as e:
                    logger.error(f"Start automation failed: {e}")
                    st.error("Failed to start automation")

with col2:
    if automated_trader:
        if st.button("⚡ Force Scan"):
            try:
                result = automated_trader.force_scan()
                if isinstance(result, dict) and result.get('success'):
                    st.success(result.get('message', 'Scan completed'))
                    st.rerun()
                else:
                    st.error(result.get('error', 'Scan failed') if isinstance(result, dict) else 'Scan failed')
            except Exception as e:
                logger.error(f"Force scan error: {e}")
                st.error("Force scan failed")

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
                    st.error("Failed to save settings")

st.divider()

# -------------------------
# Trade management tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Open Positions", "Closed Trades", "Manual Trading", "Performance Analytics"])

# ---------- Open Positions ----------
with tab1:
    st.subheader("🔓 Open Positions")

    # Interactive filters for open positions
    col_filter, col_refresh = st.columns([3, 1])
    with col_filter:
        symbol_filter = st.selectbox("Filter symbol", options=["All"], index=0, key="open_symbol_filter")
        side_filter = st.selectbox("Side", options=["All", "BUY", "SELL", "LONG", "SHORT"], index=0, key="open_side_filter")
    with col_refresh:
        refresh_interval = st.selectbox(
            "Auto-refresh",
            [0, 5, 10, 30, 60],
            index=0,
            format_func=lambda x: "Off" if x == 0 else f"Every {x} seconds",
            key="open_refresh"
        )

    try:
        is_virtual = (account_type == 'virtual')
        open_trades = db_manager.get_trades(status='open', virtual=is_virtual, exchange=current_exchange, limit=1000)

        # populate symbol filter options dynamically
        symbols = sorted({safe_str(t.symbol) for t in open_trades if getattr(t, "symbol", None)})
        symbol_filter_opts = ["All"] + symbols
        # replace previously selected if needed
        if "open_symbol_filter" in st.session_state and st.session_state.open_symbol_filter not in symbol_filter_opts:
            st.session_state.open_symbol_filter = "All"
        symbol_filter = st.selectbox("Filter symbol", options=symbol_filter_opts, index=symbol_filter_opts.index(symbol_filter) if symbol_filter in symbol_filter_opts else 0, key="open_symbol_filter")

        # Filter trades defensively
        def open_trade_matches(t):
            s_ok = (symbol_filter == "All") or (safe_str(getattr(t, "symbol", "")).upper() == str(symbol_filter).upper())
            side_val = safe_str(getattr(t, "side", ""))
            side_ok = (side_filter == "All") or (side_val.upper() == side_filter.upper())
            return s_ok and side_ok

        filtered_open = [t for t in open_trades if open_trade_matches(t)]

        if filtered_open:
            # Start websocket best-effort
            try:
                ws_symbols = list({safe_str(t.symbol) for t in filtered_open})
                trading_engine.client.start_websocket(ws_symbols)
            except Exception:
                pass

            # Build table rows
            rows = []
            for t in filtered_open:
                qty = safe_float(getattr(t, "qty", 0.0))
                entry = safe_float(getattr(t, "entry_price", 0.0))
                current_price = safe_float(trading_engine.client.get_current_price(getattr(t, "symbol", "")) or 0.0)
                pnl_val = 0.0
                if is_virtual:
                    try:
                        pnl_val = safe_float(trading_engine.calculate_virtual_pnl({
                            "symbol": getattr(t, "symbol", ""),
                            "entry_price": entry,
                            "qty": qty,
                            "side": getattr(t, "side", "")
                        }), 0.0)
                    except Exception:
                        pnl_val = safe_float(getattr(t, "pnl", 0.0))
                else:
                    pnl_val = safe_float(getattr(t, "pnl", 0.0))

                opened = getattr(t, "created_at", None)
                rows.append({
                    "ID": int(getattr(t, "id", 0) or 0),
                    "Symbol": safe_str(getattr(t, "symbol", "N/A")),
                    "Side": safe_str(getattr(t, "side", "N/A")),
                    "Quantity": f"{qty:.6f}",
                    "Entry Price": f"${entry:.6f}",
                    "Current Price": f"${current_price:.6f}",
                    "Current P&L": f"${pnl_val:.2f}",
                    "TP": f"${safe_float(getattr(t, 'tp', None)):.6f}" if getattr(t, 'tp', None) is not None else "N/A",
                    "SL": f"${safe_float(getattr(t, 'sl', None)):.6f}" if getattr(t, 'sl', None) is not None else "N/A",
                    "Opened": safe_dt(opened)
                })

            df = pd.DataFrame(rows)

            # Allow sorting by user choice
            sort_by = st.selectbox("Sort by", options=["ID", "Symbol", "Current P&L", "Opened"], index=0, key="open_sort_by")
            ascending = st.checkbox("Ascending", value=False, key="open_asc")
            if sort_by in df.columns:
                # convert P&L to numeric for sorting
                if sort_by == "Current P&L":
                    df["_pnl_sort"] = df["Current P&L"].apply(lambda x: safe_float(str(x).replace("$", "").replace(",", ""), 0.0))
                    df = df.sort_values(by="_pnl_sort", ascending=ascending).drop(columns=["_pnl_sort"])
                else:
                    df = df.sort_values(by=sort_by, ascending=ascending)

            # display as styled HTML table with colored P&L (Binance-style colors)
            def df_to_html_table(dframe: pd.DataFrame) -> str:
                headers = "".join([f"<th style='padding:6px 10px;background:#111827;color:#f3b400;border-bottom:1px solid #2b2f33'>{h}</th>" for h in dframe.columns])
                rows_html = []
                for _, r in dframe.iterrows():
                    cols = []
                    for col in dframe.columns:
                        v = r[col]
                        if col in ["Current P&L", "P&L"]:
                            v_html = highlight_pnl_html(str(v))
                            cols.append(f"<td style='padding:6px 10px;border-bottom:1px solid #2b2f33'>{v_html}</td>")
                        else:
                            cols.append(f"<td style='padding:6px 10px;border-bottom:1px solid #2b2f33;color:#e6eef8'>{v}</td>")
                    rows_html.append("<tr>" + "".join(cols) + "</tr>")
                table = f"""
                <table style='width:100%;border-collapse:collapse;margin-top:8px'>
                  <thead><tr>{headers}</tr></thead>
                  <tbody>{''.join(rows_html)}</tbody>
                </table>
                """
                return table

            st.markdown(df_to_html_table(df), unsafe_allow_html=True)

            # allow user to inspect a trade in detail
            ids = [f"{r['ID']} - {r['Symbol']}" for r in rows]
            selection = st.selectbox("Select position to inspect", options=ids, index=0, key="open_selected_position")
            if selection:
                sel_id = int(selection.split(" - ")[0])
                sel_trade = next((t for t in filtered_open if int(getattr(t, "id", 0) or 0) == sel_id), None)
                if sel_trade:
                    with st.expander("Trade Details", expanded=True):
                        st.write("**Basic Info**")
                        st.write(f"ID: {int(getattr(sel_trade, 'id', 0) or 0)}")
                        st.write(f"Symbol: {safe_str(getattr(sel_trade, 'symbol', 'N/A'))}")
                        st.write(f"Side: {safe_str(getattr(sel_trade, 'side', 'N/A'))}")
                        st.write(f"Quantity: {safe_float(getattr(sel_trade, 'qty', 0.0)):.6f}")
                        st.write(f"Entry Price: ${safe_float(getattr(sel_trade, 'entry_price', 0.0)):.6f}")
                        st.write(f"Current Price: ${safe_float(trading_engine.client.get_current_price(getattr(sel_trade, 'symbol', '')) or 0.0):.6f}")
                        st.write(f"TP: {safe_str(getattr(sel_trade, 'tp', 'N/A'))}")
                        st.write(f"SL: {safe_str(getattr(sel_trade, 'sl', 'N/A'))}")
                        st.write(f"Opened: {safe_dt(getattr(sel_trade, 'created_at', None))}")
                        st.write(f"Raw indicators: {getattr(sel_trade, 'indicators', {})}")

                        # Close position UI (keep interactive, but ask for price before action)
                        close_price_input_key = f"close_price_{sel_id}"
                        chosen_close_price = st.number_input(
                            "Close price for this trade",
                            min_value=0.000001,
                            value=float(safe_float(trading_engine.client.get_current_price(getattr(sel_trade, 'symbol', '')) or 0.0)),
                            format="%.6f",
                            key=close_price_input_key
                        )
                        if st.button("🔒 Close This Position (virtual)", key=f"close_btn_{sel_id}"):
                            try:
                                pos_id = int(getattr(sel_trade, "id", 0) or 0)
                                if trading_engine.close_virtual_trade(pos_id, float(chosen_close_price)):
                                    st.success("Position closed")
                                    st.rerun()
                                else:
                                    st.error("Failed to close position")
                            except Exception as e:
                                logger.error(f"Close position failed: {e}")
                                st.error("Failed to close position")

        else:
            st.info("No open positions")

    except Exception as e:
        logger.error(f"Error loading open positions: {e}")
        st.error(f"Error loading open positions: {e}")

    # Auto-refresh
    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()

# ---------- Closed Trades ----------
with tab2:
    st.subheader("✅ Closed Trades")
    col1, col2, col3 = st.columns(3)
    with col1:
        trade_type = st.selectbox("Trade Type", ["Virtual", "Real", "All"], index=2, key="closed_trade_type")
    with col2:
        days_back = st.selectbox("Time Period (days)", [1, 7, 30, 90], index=2, key="closed_days_back")
    with col3:
        min_pnl = st.number_input("Min P&L ($)", value=0.0, step=0.01, key="closed_min_pnl")

    try:
        is_virtual = (trade_type == "Virtual") or (trade_type == "All" and account_type == 'virtual')
        raw_closed = db_manager.get_trades(status='closed', virtual=is_virtual, exchange=current_exchange, limit=2000)

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(days_back))

        def is_within_cutoff(t):
            updated = getattr(t, "updated_at", None)
            created = getattr(t, "created_at", None)
            reference = updated if is_datetime(updated) else (created if is_datetime(created) else None)
            return reference is not None and reference >= cutoff

        trades = [t for t in raw_closed if is_within_cutoff(t)]

        # Apply min_pnl filter
        def pnl_ok(t):
            return safe_float(getattr(t, "pnl", 0.0), 0.0) >= float(min_pnl)

        trades = [t for t in trades if pnl_ok(t)]

        if trades:
            rows = []
            for t in trades[:500]:
                updated = getattr(t, "updated_at", None)
                created = getattr(t, "created_at", None)
                closed_ref = updated if is_datetime(updated) else created
                closed_str = safe_dt(closed_ref)
                duration = "N/A"
                if is_datetime(created) and is_datetime(closed_ref):
                    if closed_ref and created:
                        duration_delta = closed_ref - created
                        duration = str(duration_delta).split('.')[0]  # Remove microseconds
                    else:
                        duration = "0:00:00"  


                rows.append({
                    "ID": int(getattr(t, "id", 0) or 0),
                    "Type": "Virtual" if getattr(t, "virtual", False) else "Real",
                    "Symbol": safe_str(getattr(t, "symbol", "N/A")),
                    "Side": safe_str(getattr(t, "side", "N/A")),
                    "Quantity": f"{safe_float(getattr(t, 'qty', 0.0)):.6f}",
                    "Entry": f"${safe_float(getattr(t, 'entry_price', 0.0)):.6f}",
                    "Exit": f"${safe_float(getattr(t, 'exit_price', 0.0)):.6f}" if getattr(t, "exit_price", None) is not None else "N/A",
                    "P&L": f"${safe_float(getattr(t, 'pnl', 0.0)):.2f}",
                    "Duration": duration,
                    "Closed": closed_str
                })

            df_closed = pd.DataFrame(rows)
            # show styled HTML (colored P&L)
            def html_for_closed(dfc: pd.DataFrame) -> str:
                headers = "".join([f"<th style='padding:6px 10px;background:#111827;color:#f3b400;border-bottom:1px solid #2b2f33'>{h}</th>" for h in dfc.columns])
                rows_html = []
                for _, r in dfc.iterrows():
                    cols = []
                    for col in dfc.columns:
                        v = r[col]
                        if col == "P&L":
                            v_html = highlight_pnl_html(str(v))
                            cols.append(f"<td style='padding:6px 10px;border-bottom:1px solid #2b2f33'>{v_html}</td>")
                        else:
                            cols.append(f"<td style='padding:6px 10px;border-bottom:1px solid #2b2f33;color:#e6eef8'>{v}</td>")
                    rows_html.append("<tr>" + "".join(cols) + "</tr>")
                table = f"<table style='width:100%;border-collapse:collapse;margin-top:8px'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
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

# ---------- Manual Trading ----------
with tab3:
    st.subheader("🎯 Manual Trading")
    st.warning("⚠️ Manual trading is for virtual accounts only in this version")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Virtual Trade Entry:**")
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

        if st.button("📈 Execute Trade", type="primary"):
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
                    'indicators': {}
                }
                executed = trading_engine.execute_virtual_trade(signal)
                if executed:
                    st.success(f"Trade executed: {side} {quantity} {symbol} at ${entry_price}")
                    st.rerun()
                else:
                    st.error("Failed to execute trade")
            except Exception as e:
                logger.error(f"Error executing trade: {e}")
                st.error(f"Error executing trade: {e}")

    with col2:
        st.write("**Trade Calculator:**")
        try:
            wallet = db_manager.get_wallet_balance(account_type)
            balance = safe_float(getattr(wallet, "available", None), default=100.0) if wallet else 100.0
            st.metric("Available Balance", f"${balance:.2f}")

            risk_percent = st.slider("Risk %", 0.1, 10.0, 1.0, 0.1)
            leverage = st.slider("Leverage", 1, 20, 10)

            risk_amount = balance * (risk_percent / 100)
            position_value = risk_amount * leverage

            if entry_price > 0:
                position_size = position_value / entry_price
                st.write(f"**Risk Amount:** ${risk_amount:.2f}")
                st.write(f"**Position Value:** ${position_value:.2f}")
                st.write(f"**Position Size:** {position_size:.6f}")

                if tp_price > 0 and sl_price > 0:
                    if side.upper() in ["BUY", "LONG"]:
                        potential_profit = (tp_price - entry_price) * position_size
                        potential_loss = (entry_price - sl_price) * position_size
                    else:
                        potential_profit = (entry_price - tp_price) * position_size
                        potential_loss = (sl_price - entry_price) * position_size

                    st.write(f"**Potential Profit:** ${potential_profit:.2f}")
                    st.write(f"**Potential Loss:** ${potential_loss:.2f}")
                    if potential_loss != 0:
                        st.write(f"**R:R Ratio:** 1:{abs(potential_profit/potential_loss):.2f}")

        except Exception as e:
            logger.error(f"Error in trade calculator: {e}")
            st.error(f"Error in trade calculator: {e}")

# ---------- Performance Analytics ----------
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

        raw_trades = db_manager.get_trades(virtual=(account_type == 'virtual'), exchange=current_exchange, limit=2000)

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
                        dur_hours = (updated - created).total_seconds() / 3600 if isinstance(updated, datetime) and isinstance(created, datetime) else 0.0
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

# ----- Status footer -----
try:
    open_count = len(db_manager.get_trades(status='open', virtual=(account_type == 'virtual'), exchange=current_exchange, limit=2000))
except Exception:
    open_count = 0

st.markdown(f"""
---
**Status:** Exchange: {safe_str(current_exchange).title()} | 
Mode: {safe_str(account_type).title()} | 
Open Positions: {open_count} | 
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""", unsafe_allow_html=True)
