import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timezone, timedelta
import time
from db import DatabaseManager, WalletBalance
from logging_config import get_trading_logger

# Initialize logger
logger = get_trading_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Dashboard - AlgoTrader Pro",
    page_icon="📈",
    layout="wide"
)

# Initialize components
if 'trading_engine' not in st.session_state or 'current_exchange' not in st.session_state:
    st.error("Trading engine not initialized. Please restart the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
db_manager = DatabaseManager()
current_exchange = st.session_state.current_exchange
account_type = st.session_state.get('account_type', 'virtual')

st.title("📈 Trading Dashboard")

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
                prices[sym] = float(price) if price else "N/A"
            except Exception as e:
                prices[sym] = "N/A"
                logger.warning(f"Failed to fetch price for {sym}: {str(e)}")
                st.warning(f"Failed to fetch price for {sym}: {str(e)}")
            
            with price_cols[i]:
                value = f"${prices[sym]:.2f}" if isinstance(prices[sym], (int, float)) else prices[sym]
                st.metric(sym, value)
else:
    st.info("Select symbols to display real-time prices")

st.divider()

# Main metrics row
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

try:
    # Get wallet balance
    wallet = db_manager.get_wallet_balance(account_type, exchange=current_exchange)
    balance = wallet.available if wallet else 100.0
    
    # Get trade statistics
    stats = trading_engine.get_trade_statistics(account_type)
    
    # Get recent signals
    recent_signals = db_manager.get_signals(limit=24, exchange=current_exchange)
    
    # Get open positions
    open_trades = db_manager.get_trades(status='open', virtual=(account_type == 'virtual'), exchange=current_exchange)
    
    with col1:
        st.metric(
            f"{account_type.title()} Balance",
            f"${balance:,.2f}",
            delta=f"{stats.get('total_pnl', 0):+.2f}",
            help="Total account balance including unrealized P&L"
        )
    
    with col2:
        st.metric(
            "Open Positions",
            len(open_trades),
            delta=f"Max: {trading_engine.max_open_positions}",
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
        signals = db_manager.get_signals(limit=10, exchange=current_exchange)
        
        if signals:
            signal_data = []
            for signal in signals:
                created_at = signal.created_at
                signal_data.append({
                    'Symbol': signal.symbol,
                    'Side': signal.side,
                    'Score': f"{signal.score:.1f}",
                    'Entry': f"${signal.entry:.6f}",
                    'Market': signal.market,
                    'Created': created_at.strftime('%Y-%m-%d %H:%M') if created_at is not None else 'N/A'
                })

            
            df_signals = pd.DataFrame(signal_data)
            
            # Color code by side
            def color_side(val):
                if val.upper() in ['BUY', 'LONG']:
                    return 'color: green'
                else:
                    return 'color: red'
            
            styled_df = df_signals.style.map(color_side, subset=['Side'])
            st.dataframe(styled_df, use_container_width=True, height=300)
            
        else:
            st.info("No recent signals found")
            
    except Exception as e:
        logger.error(f"Error loading signals: {e}")
        st.error(f"Error loading signals: {e}")

with tab2:
    st.write("**Current Open Positions**")
    try:
        open_trades = db_manager.get_trades(status='open', virtual=(account_type == 'virtual'), exchange=current_exchange)
        
        if open_trades:
            trade_data = []
            for trade in open_trades:
                # Calculate current P&L
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
                    'Entry Price': f"${trade.entry_price:.6f}",
                    'Current P&L': f"${current_pnl:.2f}",
                    'Opened': created_at.strftime('%Y-%m-%d %H:%M') if created_at is not None else 'N/A'
                })

            
            df_trades = pd.DataFrame(trade_data)
            
            # Color code P&L
            def color_pnl(val):
                try:
                    val_float = float(val.replace('$', ''))
                    if val_float < 0:
                        return 'color: red'
                    elif val_float > 0:
                        return 'color: green'
                    return 'color: gray'
                except:
                    return 'color: gray'
            
            styled_df = df_trades.style.map(color_pnl, subset=['Current P&L'])
            st.dataframe(styled_df, use_container_width=True, height=300)
            
        else:
            st.info("No open positions")
            
    except Exception as e:
        logger.error(f"Error loading open positions: {e}")
        st.error(f"Error loading open positions: {e}")

with tab3:
    st.write("**Performance Analytics**")
    
    try:
        # Date range selector
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

        # Get closed trades for analysis
        all_closed_trades = db_manager.get_trades(status='closed', virtual=(account_type == 'virtual'), exchange=current_exchange)
        closed_trades = [
            t for t in all_closed_trades
            if t.updated_at is not None and start_date <= t.updated_at.date() <= end_date
        ]
        
        if closed_trades:
            # Prepare data for charts
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
                    # Cumulative P&L chart
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=trade_dates,
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='blue', width=2),
                        hovertemplate='Date: %{x}<br>P&L: $%{y:.2f}<extra></extra>'
                    ))
                    fig_cum.update_layout(
                        title="Cumulative P&L Over Time",
                        xaxis_title="Date",
                        yaxis_title="P&L ($)",
                        height=400,
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                with col2:
                    # Trade P&L distribution
                    fig_dist = px.histogram(
                        x=daily_pnl,
                        nbins=20,
                        title="Trade P&L Distribution",
                        color_discrete_sequence=['#636EFA']
                    )
                    fig_dist.update_layout(
                        xaxis_title="P&L ($)",
                        yaxis_title="Number of Trades",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Performance metrics
                st.write("**Key Performance Metrics:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Best Trade", f"${max(daily_pnl):.2f}" if daily_pnl else "$0.00")
                with col2:
                    st.metric("Worst Trade", f"${min(daily_pnl):.2f}" if daily_pnl else "$0.00")
                with col3:
                    st.metric("Avg Trade", f"${sum(daily_pnl)/len(daily_pnl):.2f}" if daily_pnl else "$0.00")
                with col4:
                    profitable_trades = len([p for p in daily_pnl if p > 0])
                    st.metric("Profitable %", f"{(profitable_trades/len(daily_pnl)*100):.1f}%" if daily_pnl else "0.0%")
        else:
            st.info("No closed trades available for the selected period")
            
    except Exception as e:
        logger.error(f"Error loading performance charts: {e}")
        st.error(f"Error loading performance charts: {e}")

st.divider()

# Market Overview
st.subheader("📊 Market Overview")

try:
    # Symbol filter
    signals = db_manager.get_signals(limit=100, exchange=current_exchange)
    if signals:
        unique_symbols = sorted(list(set(s.symbol for s in signals)))
        selected_symbol = st.selectbox("Filter by symbol", ["All"] + unique_symbols, key="market_symbol")
        
        # Filter signals
        filtered_signals = []
        for s in signals:
            symbol_val = getattr(s, "symbol", None)
            if symbol_val is not None and start_date <= s.updated_at.date() <= end_date:
                filtered_signals.append(s)

        # Count signals by symbol or show for selected
        if str(selected_symbol) == "All":
            symbol_counts = {}
            for signal in filtered_signals:
                sym = getattr(signal, "symbol", None)
                if sym is not None:
                    symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
    
            # Create bar chart
            symbols = list(symbol_counts.keys())[:10]  # Top 10
            counts = [symbol_counts[s] for s in symbols]
            
            fig_symbols = px.bar(
                x=symbols,
                y=counts,
                title="Most Active Trading Pairs (Recent Signals)",
                labels={'x': 'Symbol', 'y': 'Signal Count'},
                color=counts,
                color_continuous_scale='viridis',
                text_auto=True
            )
            fig_symbols.update_layout(height=400)
            st.plotly_chart(fig_symbols, use_container_width=True)
        else:
            # Show signal history for selected symbol
            signal_data = []
            for s in filtered_signals:
                signal_data.append({
                    'Date': s.created_at,
                    'Side': s.side,
                    'Score': s.score,
                    'Entry': s.entry
                })
            
            df = pd.DataFrame(signal_data)
            if not df.empty:
                fig = px.scatter(
                    df,
                    x="Date",
                    y="Score",
                    color="Side",
                    size="Entry",
                    title=f"Signal History for {selected_symbol}",
                    color_discrete_map={'BUY': 'green', 'SELL': 'red', 'LONG': 'green', 'SHORT': 'red'},
                    hover_data=['Entry'],
                    height=300
                )
                fig.update_traces(marker=dict(size=12, opacity=0.8))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No signals found for {selected_symbol}")
        
    else:
        st.info("No signal data available for market overview")
        
except Exception as e:
    logger.error(f"Error loading market overview: {e}")
    st.error(f"Error loading market overview: {e}")

# Auto-refresh logic
if refresh_interval > 0:
    time.sleep(refresh_interval)
    st.rerun()

# Status footer
st.markdown(f"""
---
**Status:** Trading Engine: {current_exchange.title()} | 
Trading: {'🟢 Active' if trading_engine.is_trading_enabled() else '🔴 Inactive'} | 
Mode: {account_type.title()} | 
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""")