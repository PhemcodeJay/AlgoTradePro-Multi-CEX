import streamlit as st
import pandas as pd
import asyncio
import sys
import os
from datetime import datetime, timezone
from sqlalchemy import update

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load exchange from environment
EXCHANGE = os.getenv("EXCHANGE", "binance").lower()

# Dynamically import client and engine based on exchange
if EXCHANGE == "binance":
    from binance_client import BinanceClient
    from multi_trading_engine import TradingEngine
elif EXCHANGE == "bybit":
    from bybit_client import BybitClient
    from multi_trading_engine import TradingEngine
else:
    raise ValueError(f"Unsupported exchange: {EXCHANGE}. Set EXCHANGE to 'binance' or 'bybit' in .env")

from db import db_manager, TradeModel, WalletBalance
from automated_trader import AutomatedTrader
from binance_ui_utils import calculate_portfolio_metrics
from signal_generator import get_usdt_symbols

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

st.set_page_config(
    page_title=f"Trades - AlgoTrader Pro ({EXCHANGE.capitalize()})",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache symbols to avoid repeated async calls
@st.cache_data(show_spinner=False)
def get_cached_usdt_symbols(limit: int, client_id: str) -> list:
    """Synchronous wrapper to fetch USDT symbols using async client."""
    async def fetch_symbols():
        client = st.session_state.client
        return await get_usdt_symbols(limit, client=client)
    
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(fetch_symbols())
    except Exception as e:
        logger.error(f"Error fetching cached USDT symbols for {EXCHANGE}: {e}")
        return []

# Initialize components
@st.cache_resource
def get_engine():
    return TradingEngine()

@st.cache_resource  
def get_automated_trader():
    engine = get_engine()
    return AutomatedTrader(engine)

def close_trade_safely(trade_id: str, virtual: bool = True):
    """Close a trade with proper error handling"""
    try:
        engine = st.session_state.engine
        automated_trader = get_automated_trader()
        automated_trader._ensure_session()  # Ensure active session
        
        # Get trade from database
        open_trades = [t for t in db_manager.get_trades(limit=1000) if t.status == "open"]
        trade = next((t for t in open_trades if str(t.id) == str(trade_id) or str(t.order_id) == str(trade_id)), None)
        
        if not trade:
            st.error(f"Trade {trade_id} not found on {EXCHANGE}")
            logger.error(f"Trade {trade_id} not found on {EXCHANGE}")
            return False
        
        # Get current price for PnL calculation
        loop = asyncio.get_event_loop()
        current_price = loop.run_until_complete(st.session_state.client.get_current_price(trade.symbol))  # type: ignore
        # Calculate PnL
        pnl = engine.calculate_virtual_pnl(trade.to_dict()) if virtual else trade.pnl or 0
        
        # Update trade in database
        try:
            db_manager.session.execute(
                update(TradeModel)
                .where(TradeModel.order_id == trade.order_id)
                .values(
                    status="closed",
                    exit_price=current_price,
                    pnl=pnl,
                    closed_at=datetime.now(timezone.utc)
                )
            )
            db_manager.session.commit()
            success = True
        except Exception as e:
            db_manager.session.rollback()
            st.error(f"Database error updating trade {trade.order_id} on {EXCHANGE}: {e}")
            logger.error(f"Database error updating trade {trade.order_id} on {EXCHANGE}: {e}", exc_info=True)
            success = False
        
        if success:
            # Update balance if virtual trade
            if virtual:
                engine.update_virtual_balances(pnl)
            else:
                engine.sync_real_balance()
            logger.info(f"Trade {trade_id} closed successfully on {EXCHANGE}")
            return True
        return False
    except Exception as e:
        st.error(f"Error closing trade {trade_id} on {EXCHANGE}: {e}")
        logger.error(f"Error closing trade {trade_id} on {EXCHANGE}: {e}")
        return False

def display_manual_trading():
    """Manual trading interface"""
    st.subheader("📈 Manual Trading")
    col1, col2 = st.columns([1, 2])
    with col1:
        client = st.session_state.client
        client_id = str(id(client))
        symbols = get_cached_usdt_symbols(50, client_id)  # Use cached wrapper
        symbol = st.selectbox("Symbol", symbols, key="manual_symbol")
        side = st.selectbox("Side", ["Buy", "Sell"], key="manual_side")
        qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1, key="manual_qty")
        price = st.number_input("Price", min_value=0.0, value=0.0, step=0.1, key="manual_price")
        leverage = st.number_input("Leverage", min_value=1, max_value=125, value=10, key="manual_leverage")
        
        if st.button("Execute Manual Trade", key="execute_manual"):
            if not symbol or qty <= 0:
                st.error("Invalid symbol or quantity")
                return
            
            engine = st.session_state.engine
            loop = asyncio.get_event_loop()  # Define loop here
            trade_data = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": price if price > 0 else loop.run_until_complete(engine.client.get_current_price(symbol)),  # type: ignore
                "order_id": f"manual_{symbol}_{int(datetime.now().timestamp())}",
                "virtual": st.session_state.get("trading_mode", "virtual") == "virtual",
                "status": "open",
                "leverage": leverage,
                "strategy": "Manual"
            }
            
            try:
                success = engine.db.add_trade(trade_data)
                if success:
                    if trade_data["virtual"]:
                        st.success(f"✅ Virtual trade executed: {symbol} {side}")
                    else:
                        order_result = loop.run_until_complete(engine.client.place_order(
                            symbol=symbol,
                            side=side,
                            qty=qty,
                            price=trade_data["entry_price"],
                            order_type="Market"
                        ))  # type: ignore
                        if order_result.get("success"):
                            engine.sync_real_balance()
                            st.success(f"✅ Real trade executed: {symbol} {side}")
                        else:
                            st.error(f"❌ Failed to execute real trade: {order_result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error executing manual trade: {e}")
                logger.error(f"Error executing manual trade: {e}")

    with col2:
        st.write("**Open Trades**")
        open_trades = [t for t in db_manager.get_trades(limit=100) if t.status == "open"]
        if open_trades:
            trade_data = []
            loop = asyncio.get_event_loop()
            for t in open_trades:
                current_price = loop.run_until_complete(st.session_state.client.get_current_price(t.symbol))  # type: ignore
                unreal_pnl = (current_price - t.entry_price) * t.qty if t.side.upper() == "BUY" else (t.entry_price - current_price) * t.qty
                trade_data.append({
                    "Symbol": t.symbol,
                    "Side": t.side,
                    "Qty": t.qty,
                    "Entry Price": f"${t.entry_price:.4f}",
                    "Current Price": f"${current_price:.4f}",
                    "Unrealized PnL": f"${unreal_pnl:.2f}",
                    "Order ID": t.order_id
                })
            st.dataframe(pd.DataFrame(trade_data))
            
            selected_trade = st.selectbox("Select Trade to Close", [t.order_id for t in open_trades], key="close_trade")
            if st.button("Close Trade", key="close_trade_button"):
                success = close_trade_safely(selected_trade, virtual=st.session_state.get("trading_mode", "virtual") == "virtual")
                if success:
                    st.success(f"Trade {selected_trade} closed")
                else:
                    st.error(f"Failed to close trade {selected_trade}")

def get_or_create_event_loop():
    """Return an existing event loop, or create a new one if none exists."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:  # no loop in current thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def display_automation_tab():
    """Automation interface"""
    st.subheader("🤖 Trading Automation")
    automated_trader = get_automated_trader()
    loop = get_or_create_event_loop()

    col1, col2 = st.columns(2)

    # --- Left Column: Settings & Controls ---
    with col1:
        st.write("**Automation Settings**")
        min_score = st.slider("Minimum Signal Score", 30, 90, 50, key="auto_min_score")
        max_positions = st.slider("Max Open Positions", 1, 20, 5, key="auto_max_positions")

        if st.button("Apply Settings", key="apply_auto_settings"):
            automated_trader.min_signal_score = min_score
            automated_trader.max_positions = max_positions
            st.success("Automation settings applied")

        st.write("**Enable / Control Automation**")
        automation_enabled = st.checkbox("Enable Automated Trading", value=False, key="automation_enabled")

        # Start Automation
        if st.button("Start Automation", key="start_auto"):
            if automation_enabled:
                if not automated_trader.is_running:
                    try:
                        loop.create_task(automated_trader.start())
                        st.success("Automation started")
                    except Exception as e:
                        st.error(f"Error starting automation: {e}")
                        logger.error(f"Error starting automation: {e}", exc_info=True)
                else:
                    st.info("Automation already running")
            else:
                st.warning("Please enable automated trading first")

        # Stop Automation
        if st.button("Stop Automation", key="stop_auto"):
            if automated_trader.is_running:
                try:
                    loop.create_task(automated_trader.stop())
                    st.success("Automation stopped")
                except Exception as e:
                    st.error(f"Error stopping automation: {e}")
                    logger.error(f"Error stopping automation: {e}", exc_info=True)
            else:
                st.info("Automation is not running")

    # --- Right Column: Status ---
    with col2:
        st.write("**Automation Status**")
        try:
            status = loop.run_until_complete(automated_trader.get_status())
            st.metric("Enabled", "Yes" if automation_enabled else "No")
            st.metric("Running Trades", status["current_positions"])
            st.metric("Total Executed", status["stats"]["trades_executed"])
        except Exception as e:
            st.error(f"Error fetching automation status: {e}")
            logger.error(f"Error fetching automation status: {e}", exc_info=True)

            
def main():
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">💼 Trades ({EXCHANGE.capitalize()})</h1>
        <p style="color: #888; margin: 0;">Manage Your Trading Activity</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize engine and client
    if not st.session_state.get("engine"):
        st.session_state.engine = get_engine()
        logger.info(f"{EXCHANGE.capitalize()} trading engine initialized")
    if not st.session_state.get("client"):
        client_class = BinanceClient if EXCHANGE == "binance" else BybitClient
        st.session_state.client = client_class()
        logger.info(f"{EXCHANGE.capitalize()} client initialized")

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Trading Controls")
        trading_mode = st.selectbox(
            "Trading Mode",
            ["virtual", "real"],
            index=0 if st.session_state.get('trades_trading_mode', 'virtual') == 'virtual' else 1,
            key="trades_trading_mode"
        )
        st.session_state.trading_mode = trading_mode
        
        if st.button("📊 Back to Dashboard", key="back_to_dashboard"):
            st.switch_page("app.py")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Open Trades",
        "📜 Trading History",
        "🖐️ Manual Trading",
        "🤖 Automation",
        "📊 Statistics"
    ])

    with tab1:
        st.subheader("📈 Open Trades")
        open_trades = [t for t in db_manager.get_trades(limit=100) if t.status == "open"]
        if open_trades:
            trade_data = []
            loop = asyncio.get_event_loop()
            for t in open_trades:
                current_price = loop.run_until_complete(st.session_state.client.get_current_price(t.symbol))  # type: ignore
                unreal_pnl = (current_price - t.entry_price) * t.qty if t.side.upper() == "BUY" else (t.entry_price - current_price) * t.qty
                trade_data.append({
                    "Symbol": t.symbol,
                    "Side": t.side,
                    "Qty": t.qty,
                    "Entry Price": f"${t.entry_price:.4f}",
                    "Current Price": f"${current_price:.4f}",
                    "Unrealized PnL": f"${unreal_pnl:.2f}",
                    "Order ID": t.order_id
                })
            st.dataframe(pd.DataFrame(trade_data))
        else:
            st.info("No open trades")

    with tab2:
        st.subheader("📜 Trading History")
        closed_trades = [t for t in db_manager.get_trades(limit=100) if t.status == "closed"]
        if closed_trades:
            history_data = []
            for t in closed_trades:
                history_data.append({
                    "Symbol": t.symbol,
                    "Side": t.side,
                    "Qty": t.qty,
                    "Entry Price": f"${t.entry_price:.4f}",
                    "Exit Price": f"${t.exit_price:.4f}" if t.exit_price else "N/A",
                    "PnL": f"${t.pnl:.2f}" if t.pnl is not None else "N/A",
                    "Closed At": t.closed_at.isoformat()[:19] if t.closed_at else "N/A",
                    "Status": "✅" if t.pnl and t.pnl > 0 else "❌" if t.pnl and t.pnl < 0 else "➖"
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, height=500)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Export Trading History",
                csv,
                "trading_history.csv",
                "text/csv"
            )
        else:
            st.info(f"No trading history available on {EXCHANGE}. Start trading to see your history here!")
    
    with tab3:
        display_manual_trading()
    
    with tab4:
        display_automation_tab()
    
    with tab5:
        st.subheader("📊 Trading Statistics")
        
        engine = st.session_state.engine
        all_trades = engine.get_closed_virtual_trades() + engine.get_closed_real_trades()
        
        if all_trades:
            metrics = calculate_portfolio_metrics([t.to_dict() for t in all_trades])
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Total Trades", metrics['total_trades'])
            
            with metric_col2:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            
            with metric_col3:
                st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
            
            with metric_col4:
                st.metric("Avg P&L/Trade", f"${metrics['avg_pnl']:.2f}")
            
            st.markdown("### 🎯 Detailed Statistics")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.metric("Profitable Trades", metrics['profitable_trades'])
                st.metric("Best Trade", f"${metrics['best_trade']:.2f}")
            
            with detail_col2:
                losing_trades = metrics['total_trades'] - metrics['profitable_trades']
                st.metric("Losing Trades", losing_trades)
                st.metric("Worst Trade", f"${metrics['worst_trade']:.2f}")
            
            st.markdown("### 📈 Performance by Symbol")
            
            symbol_performance = {}
            for trade in all_trades:
                symbol = trade.symbol
                pnl = trade.pnl or 0
                
                if symbol not in symbol_performance:
                    symbol_performance[symbol] = {'trades': 0, 'total_pnl': 0}
                
                symbol_performance[symbol]['trades'] += 1
                symbol_performance[symbol]['total_pnl'] += pnl
            
            if symbol_performance:
                symbol_data = []
                for symbol, data in symbol_performance.items():
                    symbol_data.append({
                        "Symbol": symbol,
                        "Trades": data['trades'],
                        "Total PnL": f"${data['total_pnl']:.2f}",
                        "Avg PnL": f"${data['total_pnl'] / data['trades']:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(symbol_data))
        else:
            st.info(f"No trading statistics available on {EXCHANGE}. Complete some trades to see detailed analytics!")


    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#888;'> AlgoTrader Pro - Multi Exchange (Binance & Bybit) v2.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exchange: {EXCHANGE.capitalize()}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()