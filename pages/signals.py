import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from datetime import datetime
import asyncio
from ml import MLFilter

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
    from multi_trading_engine import TradingEngine  # Use same engine for Bybit
else:
    raise ValueError(f"Unsupported exchange: {EXCHANGE}. Set EXCHANGE to 'binance' or 'bybit' in .env")

from db import db_manager
from indicators import get_candles
from signal_generator import generate_signals, get_usdt_symbols, analyze_single_symbol
from notification_pdf import send_all_notifications

# Configure logging
from logging_config import get_logger
logger = get_logger(__name__)

st.set_page_config(page_title=f"Signals - AlgoTrader Pro ({EXCHANGE.capitalize()})", page_icon="🎯", layout="wide")

# Cache symbols to avoid repeated async calls
@st.cache_data(show_spinner=False)
def get_cached_usdt_symbols(limit: int, client_id: str) -> list:
    """Synchronous wrapper to fetch USDT symbols using async client."""
    async def fetch_symbols():
        client = st.session_state.client
        return await get_usdt_symbols(limit, client=client)
    
    try:
        return asyncio.run(fetch_symbols())
    except Exception as e:
        logger.error(f"Error fetching cached USDT symbols for {EXCHANGE}: {e}")
        return []

# Cache single symbol analysis to avoid repeated async calls
@st.cache_data(show_spinner=False)
def analyze_symbol_cached(symbol: str, interval: str, client_id: str) -> dict:
    """Synchronous wrapper for analyze_single_symbol."""
    async def analyze():
        client = st.session_state.client
        return await analyze_single_symbol(symbol, interval, client=client)
    
    try:
        return asyncio.run(analyze())
    except Exception as e:
        logger.error(f"Error analyzing {symbol} on {EXCHANGE}: {e}")
        return {}

def create_signal_chart(signal_data):
    """Create a candlestick chart with entry, TP, and SL lines"""
    try:
        symbol = signal_data.get('Symbol', signal_data.get('symbol', 'BTCUSDT'))
        try:
            entry = float(signal_data.get('Entry', signal_data.get('entry', 0)) or 0)
            tp = float(signal_data.get('TP', signal_data.get('tp', 0)) or 0)
            sl = float(signal_data.get('SL', signal_data.get('sl', 0)) or 0)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid price values for {symbol} on {EXCHANGE}: {e}")
            return None

        # Get candlestick data
        candles = get_candles(symbol, "60", limit=50)
        if not candles or not isinstance(candles, list) or len(candles) == 0:
            logger.warning(f"No candlestick data for {symbol} on {EXCHANGE}")
            return None

        df = pd.DataFrame(candles)
        required_columns = ['time', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Invalid candlestick data format for {symbol} on {EXCHANGE}: missing columns")
            return None

        df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        if df['time'].isna().any():
            logger.error(f"Invalid timestamp data for {symbol} on {EXCHANGE}")
            return None

        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
        
        # Add signal lines
        if entry > 0:
            fig.add_hline(y=entry, line_dash="dash", line_color="blue", 
                         annotation_text=f"Entry: ${entry:.4f}")
        if tp > 0:
            fig.add_hline(y=tp, line_dash="dot", line_color="green", 
                         annotation_text=f"Take Profit: ${tp:.4f}")
        if sl > 0:
            fig.add_hline(y=sl, line_dash="dot", line_color="red", 
                         annotation_text=f"Stop Loss: ${sl:.4f}")
        
        fig.update_layout(
            title=f"{symbol} - Technical Analysis ({EXCHANGE.capitalize()})",
            yaxis_title="Price (USDT)",
            xaxis_title="Time",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating signal chart for {symbol} on {EXCHANGE}: {e}")
        return None

def display_signal_details(signal):
    """Display detailed signal information"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Signal Details**")
        st.write(f"**Symbol:** {signal.get('Symbol', signal.get('symbol', 'N/A'))}")
        st.write(f"**Side:** {signal.get('Side', signal.get('side', 'N/A'))}")
        st.write(f"**Type:** {signal.get('signal_type', 'N/A')}")
        st.write(f"**Score:** {signal.get('Score', '0%')}")
    
    with col2:
        st.markdown("**💰 Price Levels**")
        st.write(f"**Entry:** ${float(signal.get('Entry', signal.get('entry', 0)) or 0):.4f}")
        st.write(f"**Take Profit:** ${float(signal.get('TP', signal.get('tp', 0)) or 0):.4f}")
        st.write(f"**Stop Loss:** ${float(signal.get('SL', signal.get('sl', 0)) or 0):.4f}")
        st.write(f"**Trail Stop:** ${float(signal.get('Trail', signal.get('trail', 0)) or 0):.4f}")
    
    with col3:
        st.markdown("**⚙️ Trading Info**")
        st.write(f"**Market:** {signal.get('Market', signal.get('market', 'N/A'))}")
        st.write(f"**BB Slope:** {signal.get('bb_slope', 'N/A')}")
        st.write(f"**Leverage:** {signal.get('leverage', 10)}x")
        created_at = signal.get('Time', signal.get('created_at', 'N/A'))
        st.write(f"**Generated:** {created_at if created_at != 'N/A' else 'Unknown'}")

async def execute_real_trade(engine, signal):
    """Execute a real trade based on a signal"""
    try:
        symbol = signal.get("symbol")
        if not isinstance(symbol, str):
            logger.error(f"Symbol must be a non-empty string on {EXCHANGE}")
            return False
        
        side = signal.get("side", "Buy")
        entry_price = signal.get("entry")
        if not entry_price:
            entry_price = await engine.client.get_current_price(symbol)
        if entry_price <= 0:
            logger.error(f"Invalid entry price for {symbol} on {EXCHANGE}")
            return False
        
        position_size = engine.calculate_position_size(symbol, entry_price)
        trade_data = {
            "symbol": symbol,
            "side": side,
            "qty": position_size,
            "entry_price": entry_price,
            "order_id": f"real_{symbol}_{int(datetime.now().timestamp())}",
            "virtual": False,
            "status": "open",
            "score": signal.get("score"),
            "strategy": signal.get("strategy", "Auto"),
            "leverage": signal.get("leverage", 10)
        }
        order_result = await engine.client.place_order(
            symbol=symbol,
            side=side,
            qty=trade_data["qty"],
            price=entry_price,
            order_type="Market"
        )
        if order_result.get("success"):
            engine.db.add_trade(trade_data)
            engine.sync_real_balance()
            logger.info(f"Real trade executed for {symbol} on {EXCHANGE}")
            return True
        else:
            logger.error(f"Real trade failed on {EXCHANGE}: {order_result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error executing real trade on {EXCHANGE}: {e}")
        return False

async def execute_single_symbol_real_trade(engine, signal):
    """Execute a real trade for single symbol analysis"""
    try:
        symbol = signal.get("symbol")
        if not isinstance(symbol, str):
            logger.error(f"Invalid symbol for {EXCHANGE}. Cannot execute trade.")
            return False
        
        side = signal.get("side", "Buy")
        entry_price = signal.get("entry")
        if not entry_price:
            entry_price = await engine.client.get_current_price(symbol)
        if not entry_price or entry_price <= 0:
            logger.error(f"Invalid entry price for {symbol} on {EXCHANGE}. Cannot execute trade.")
            return False
        
        position_size = engine.calculate_position_size(symbol, entry_price)
        trade_data = {
            "symbol": symbol,
            "side": side,
            "qty": position_size,
            "entry_price": entry_price,
            "order_id": f"real_{symbol}_{int(datetime.now().timestamp())}",
            "virtual": False,
            "status": "open",
            "score": signal.get("score"),
            "strategy": signal.get("strategy", "Auto"),
            "leverage": signal.get("leverage", 10)
        }
        order_result = await engine.client.place_order(
            symbol=symbol,
            side=side,
            qty=trade_data["qty"],
            price=entry_price,
            order_type="Market"
        )
        if order_result.get("success"):
            engine.db.add_trade(trade_data)
            engine.sync_real_balance()
            logger.info(f"Real trade executed for {symbol} on {EXCHANGE}")
            return True
        else:
            logger.error(f"Real trade failed on {EXCHANGE}: {order_result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error executing real trade for {symbol} on {EXCHANGE}: {e}")
        return False

def main():
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #00ff88; margin-bottom: 2rem;">
        <h1 style="color: #00ff88; margin: 0;">🎯 Trading Signals ({EXCHANGE.capitalize()})</h1>
        <p style="color: #888; margin: 0;">AI-Powered Signal Generation & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize engine and client from session state
    if not st.session_state.get("engine"):
        st.session_state.engine = TradingEngine()
        logger.info(f"{EXCHANGE.capitalize()} trading engine initialized")
    engine = st.session_state.engine

    if not st.session_state.get("client"):
        client_class = BinanceClient if EXCHANGE == "binance" else BybitClient
        st.session_state.client = client_class()
        logger.info(f"{EXCHANGE.capitalize()} client initialized")
    client = st.session_state.client
    client_id = str(id(client))  # Convert to str for caching

    # Initialize session state
    if 'generated_signals' not in st.session_state:
        st.session_state.generated_signals = []
        try:
            # Fetch recent signals from DB as fallback
            db_signals = db_manager.get_signals(limit=10)
            st.session_state.generated_signals = [s.to_dict() for s in db_signals if s.score >= 40]
            logger.info(f"Initialized {len(st.session_state.generated_signals)} signals from database for {EXCHANGE}")
        except Exception as e:
            logger.error(f"Error fetching initial signals from DB for {EXCHANGE}: {e}")
            st.session_state.generated_signals = []
    
    if 'signal_generation_in_progress' not in st.session_state:
        st.session_state.signal_generation_in_progress = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Signal Controls")
        
        # Trading mode selection
        trading_mode = st.selectbox(
            "Trading Mode", 
            ["virtual", "real"], 
            index=0 if st.session_state.get('trading_mode', 'virtual') == 'virtual' else 1,
            key="signals_trading_mode"
        )
        st.session_state.trading_mode = trading_mode
        
        st.divider()
        
        # Signal generation settings
        st.subheader("📊 Generation Settings")
        top_n_signals = st.slider("Number of Signals", 1, 20, 10, key="top_n_signals")
        min_score = st.slider("Minimum Score", 30, 90, 50, key="min_score")
        
        # Symbol selection
        available_symbols = get_cached_usdt_symbols(100, client_id)
        selected_symbols = st.multiselect(
            "Select Symbols (leave empty for auto-selection)",
            available_symbols,
            default=[],
            key="selected_symbols"
        )
        
        st.divider()
        
        # Generation controls
        if st.button("🎯 Generate New Signals", 
                     disabled=st.session_state.signal_generation_in_progress,
                     key="generate_signals"):
            st.session_state.signal_generation_in_progress = True
        
        if st.button("📤 Send Notifications", 
                     disabled=len(st.session_state.generated_signals) == 0,
                     key="send_notifications"):
            if st.session_state.generated_signals:
                try:
                    send_all_notifications(st.session_state.generated_signals)
                    st.success("Notifications sent!")
                except Exception as e:
                    st.error(f"Notification error for {EXCHANGE}: {e}")
                    logger.error(f"Notification error for {EXCHANGE}: {e}")
        
        st.divider()
        
        # Filters for database signals
        st.subheader("🔍 Database Filters")
        symbol_filter = st.text_input("Symbol Filter", placeholder="BTC, ETH, etc.", key="symbol_filter")
        side_filter = st.selectbox("Side Filter", ["All", "Buy", "Sell"], key="side_filter")
        
        if st.button("📊 Back to Dashboard", key="back_to_dashboard"):
            st.switch_page("app.py")
    
    # Handle signal generation
    if st.session_state.signal_generation_in_progress:
        with st.spinner(f"🔄 Generating signals for {EXCHANGE.capitalize()}... This may take a few minutes."):
            try:
                symbols_to_scan = selected_symbols if selected_symbols else get_cached_usdt_symbols(50, client_id)
                signals = asyncio.run(generate_signals(
                    symbols_to_scan, 
                    interval="60", 
                    top_n=top_n_signals,
                    trading_mode=trading_mode,
                    client=client
                ))
                filtered_signals = [s for s in signals if float(str(s.get('Score', '0%')).replace('%', '')) >= min_score]
                st.session_state.generated_signals = filtered_signals
                st.success(f"✅ Generated {len(filtered_signals)} signals, saved {len(filtered_signals)} to database for {EXCHANGE}")
            except Exception as e:
                st.error(f"Error generating signals for {EXCHANGE}: {e}")
                logger.error(f"Signal generation error for {EXCHANGE}: {e}")
            finally:
                st.session_state.signal_generation_in_progress = False
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🆕 Generated Signals", "💾 Database Signals", "🔍 Single Symbol Analysis", "🤖 ML Signal Filter"])
    
    with tab1:
        st.subheader("🆕 Recently Generated Signals")
        logger.debug(f"Rendering Tab 1 with {len(st.session_state.generated_signals)} signals for {EXCHANGE}")
        signals = st.session_state.generated_signals

        if signals:
            signals_data = []
            for s in signals:
                try:
                    entry_val = float(s.get("Entry", s.get("entry", 0)) or 0)
                    tp_val = float(s.get("TP", s.get("tp", 0)) or 0)
                    sl_val = float(s.get("SL", s.get("sl", 0)) or 0)

                    # Normalize score
                    raw_score = s.get("Score") or s.get("score") or 0
                    score_val = float(str(raw_score).replace("%", "") or 0)

                    signals_data.append({
                        "Symbol": s.get("Symbol", s.get("symbol", "N/A")),
                        "Side": s.get("Side", s.get("side", "N/A")),
                        "Score": f"{score_val:.2f}%",
                        "Entry": f"${entry_val:.4f}",
                        "Take Profit": f"${tp_val:.4f}",
                        "Stop Loss": f"${sl_val:.4f}",
                        "Market": s.get("Market", s.get("market", "N/A")),
                        "Time": s.get("Time", s.get("created_at", 'N/A'))
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid signal for {EXCHANGE}: {s}, error: {e}")
                    continue

            if signals_data:
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, height=400)

                selected_idx = st.selectbox(
                    "Select signal for detailed analysis:",
                    range(len(signals)),
                    format_func=lambda idx: f"{signals[idx].get('Symbol', signals[idx].get('symbol', 'N/A'))} - {signals[idx].get('Score', '0%')}",
                    key="select_signal"
                )

                if selected_idx is not None:
                    selected_signal = signals[selected_idx]
                    display_signal_details(selected_signal)
                    chart = create_signal_chart(selected_signal)
                    if chart:
                        st.plotly_chart(chart)
                    else:
                        st.error(f"Failed to generate chart for selected signal on {EXCHANGE}.")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📈 Execute Virtual Trade", key="virtual_trade_tab1"):
                            try:
                                success = engine.execute_virtual_trade(selected_signal, trading_mode="virtual")
                                if success:
                                    st.success("✅ Virtual trade executed!")
                                else:
                                    st.error("❌ Failed to execute virtual trade")
                            except Exception as e:
                                st.error(f"Error executing virtual trade for {EXCHANGE}: {e}")
                                logger.error(f"Error executing virtual trade for {EXCHANGE}: {e}")
                    with col2:
                        real_disabled = st.session_state.get("trading_mode", "virtual") != "real"
                        if st.button("💰 Execute Real Trade", disabled=real_disabled, key="execute_real_trade_tab1"):
                            if real_disabled:
                                st.info("Switch to real mode to execute real trades")
                            else:
                                try:
                                    if not engine.is_trading_enabled():
                                        st.error("Trading disabled or emergency stop active")
                                    else:
                                        success = asyncio.run(execute_real_trade(engine, selected_signal))
                                        if success:
                                            st.success(f"✅ Real trade executed for {selected_signal.get('symbol', 'N/A')} on {EXCHANGE}")
                                        else:
                                            st.error(f"❌ Failed to execute real trade on {EXCHANGE}")
                                except Exception as e:
                                    st.error(f"Error executing real trade for {EXCHANGE}: {e}")
                                    logger.error(f"Error executing real trade for {EXCHANGE}: {e}")
            else:
                st.warning("No valid signals available to display.")
        else:
            st.info(f"🎯 Click 'Generate New Signals' to start analyzing the markets on {EXCHANGE}!")

    with tab2:
        st.subheader("💾 Historical Signals from Database")
        try:
            db_signals = db_manager.get_signals(limit=50)
            if db_signals:
                filtered_db_signals = []
                for signal in db_signals:
                    signal_dict = signal.to_dict()
                    if symbol_filter and symbol_filter.upper() not in signal_dict.get('symbol', '').upper():
                        continue
                    if side_filter != "All" and signal_dict.get('side', '').lower() != side_filter.lower():
                        continue
                    if signal_dict.get('score', 0) < min_score:
                        continue
                    filtered_db_signals.append(signal_dict)
                
                if filtered_db_signals:
                    db_data = []
                    for s in filtered_db_signals[:30]:
                        score_val = s.get('score', 0)
                        entry_val = s.get('entry', 0)
                        created_val = s.get("created_at", None)
                        score_str = f"{float(score_val or 0):.1f}%" if score_val is not None else "0.0%"
                        entry_str = f"${float(entry_val or 0):.4f}" if entry_val is not None else "$0.0000"
                        created_str = str(created_val)[:19] if created_val is not None else "N/A"
                        db_data.append({
                            "Symbol": s.get("symbol", "N/A"),
                            "Side": s.get("side", "N/A"),
                            "Score": score_str,
                            "Entry": entry_str,
                            "Strategy": s.get("strategy", "N/A"),
                            "Interval": s.get("interval", "N/A"),
                            "Created": created_str
                        })
                    
                    db_df = pd.DataFrame(db_data)
                    st.dataframe(db_df, height=400)
                    
                    csv = db_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Signals CSV",
                        csv,
                        "trading_signals.csv",
                        "text/csv",
                        key="download_signals"
                    )
                else:
                    st.info("No signals match the current filters")
            else:
                st.info("No signals in database. Generate some signals first!")
        except Exception as e:
            st.error(f"Error loading database signals for {EXCHANGE}: {e}")
            logger.error(f"Database signals error for {EXCHANGE}: {e}")

    with tab3:
        st.subheader("🔍 Single Symbol Analysis")
        col1, col2 = st.columns([1, 2])
        with col1:
            analysis_symbol = st.selectbox(
                "Select Symbol for Analysis:",
                get_cached_usdt_symbols(50, client_id),
                key="analysis_symbol"
            )
            analysis_interval = st.selectbox(
                "Time Interval:",
                ["15", "30", "60", "240", "D"],
                index=2,
                key="analysis_interval"
            )
            if st.button("🔍 Analyze Symbol", key="analyze_symbol"):
                if analysis_symbol:
                    with st.spinner(f"Analyzing {analysis_symbol} on {EXCHANGE}..."):
                        try:
                            analysis_result = analyze_symbol_cached(analysis_symbol, analysis_interval, client_id)
                            if analysis_result:
                                st.session_state['single_analysis'] = analysis_result
                                st.success(f"✅ Analysis complete for {analysis_symbol} on {EXCHANGE}")
                            else:
                                st.warning(f"No significant signal found for {analysis_symbol} on {EXCHANGE}")
                        except Exception as e:
                            st.error(f"Analysis error for {EXCHANGE}: {e}")
                            logger.error(f"Analysis error for {EXCHANGE}: {e}")
                else:
                    st.warning("Please select a symbol to analyze")
        with col2:
            if 'single_analysis' in st.session_state:
                analysis = st.session_state['single_analysis']
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Signal Score", f"{analysis.get('score', 0):.1f}%")
                    st.metric("Signal Type", analysis.get('signal_type', 'N/A').title())
                with metrics_col2:
                    indicators = analysis.get('indicators', {})
                    st.metric("RSI", f"{indicators.get('rsi', 0):.1f}")
                    st.metric("Price", f"${indicators.get('price', 0):.4f}")
                with metrics_col3:
                    st.metric("Side", analysis.get('side', 'N/A'))
                    st.metric("Volatility", f"{indicators.get('volatility', 0):.2f}%")
                chart = create_signal_chart(analysis)
                if chart:
                    st.plotly_chart(chart)
            else:
                st.info(f"Select a symbol and click 'Analyze Symbol' to see detailed analysis on {EXCHANGE}")
        
        trade_col1, trade_col2 = st.columns(2)
        with trade_col1:
            virtual_disabled = 'single_analysis' not in st.session_state
            if st.button("💻 Execute Virtual Trade", disabled=virtual_disabled, key="virtual_trade_tab3"):
                try:
                    success = engine.execute_virtual_trade(
                        st.session_state['single_analysis'], trading_mode="virtual"
                    )
                    if success:
                        st.success(f"✅ Virtual trade executed for {analysis_symbol} on {EXCHANGE}")
                    else:
                        st.error(f"❌ Failed to execute virtual trade on {EXCHANGE}")
                except Exception as e:
                    st.error(f"Virtual trade error for {EXCHANGE}: {e}")
                    logger.error(f"Virtual trade error for {EXCHANGE}: {e}")
        with trade_col2:
            real_disabled = 'single_analysis' not in st.session_state or trading_mode != "real"
            if st.button("💰 Execute Real Trade", disabled=real_disabled, key="execute_real_trade_tab3"):
                try:
                    if not engine.is_trading_enabled():
                        st.error("Trading is disabled or emergency stop is active. Cannot execute trade.")
                    else:
                        signal = st.session_state['single_analysis']
                        success = asyncio.run(execute_single_symbol_real_trade(engine, signal))
                        if success:
                            st.success(f"✅ Real trade executed for {signal.get('symbol', 'N/A')} on {EXCHANGE}")
                        else:
                            st.error(f"❌ Failed to execute real trade on {EXCHANGE}")
                except Exception as e:
                    st.error(f"Real trade error for {EXCHANGE}: {e}")
                    logger.error(f"Real trade error for {EXCHANGE}: {e}")

    with tab4:
        st.subheader("🤖 ML-Powered Signal Filtering & Live Scoring")
        ml_filter = MLFilter()
        threshold = st.slider(
            "ML Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Minimum ML probability for signal to pass filter",
            key="ml_threshold"
        )
        signals = db_manager.get_signals(limit=100)
        st.write(f"Fetched {len(signals)} signals from database for {EXCHANGE}")
        signals_dicts = [s.to_dict() for s in signals]
        filtered_signals = ml_filter.filter_signals(signals_dicts, threshold=threshold)
        st.write(f"{len(filtered_signals)} signals passed the ML filter")
        st.dataframe(pd.DataFrame(filtered_signals))
        st.markdown("---")
        if st.button("Show Feature Importance", key="feature_importance"):
            importance = ml_filter.get_feature_importance()
            if importance:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
                st.bar_chart(importance_df)
            else:
                st.info("No trained ML model available")
        st.markdown("---")
        st.markdown("### Train ML Model from Trades Database")
        if st.button("Train Model from Trades", key="train_model"):
            trades = db_manager.get_trades(limit=1000)
            training_data = []
            for trade in trades:
                # Fetch corresponding signal for the trade
                signals = db_manager.get_signals(limit=1)
                signal = next((s for s in signals if s.symbol == trade.symbol and abs((s.created_at - trade.timestamp).total_seconds()) < 3600), None)
                if signal and signal.indicators:
                    training_data.append({
                        'indicators': signal.indicators,
                        'profit': trade.pnl if trade.pnl is not None else 0
                    })
            if not training_data:
                st.error(f"No trades with matching signal indicators found for training on {EXCHANGE}")
            else:
                success = ml_filter.train_model(training_data)
                if success:
                    st.success("ML model trained successfully")
                else:
                    st.error(f"Failed to train ML model on {EXCHANGE}. Check logs for details")
                    logger.error(f"Failed to train ML model on {EXCHANGE}")
        st.markdown("---")
        st.markdown("### Live Signal Scoring (Auto)")
        rsi = st.number_input("RSI", value=50.0, key="rsi_input")
        macd = st.number_input("MACD", value=0.0, key="macd_input")
        macd_signal = st.number_input("MACD Signal", value=0.0, key="macd_signal_input")
        macd_hist = st.number_input("MACD Histogram", value=0.0, key="macd_hist_input")
        bb_upper = st.number_input("BB Upper", value=0.0, key="bb_upper_input")
        bb_middle = st.number_input("BB Middle", value=0.0, key="bb_middle_input")
        bb_lower = st.number_input("BB Lower", value=0.0, key="bb_lower_input")
        price = st.number_input("Price", value=0.0, key="price_input")
        volume_ratio = st.number_input("Volume Ratio", value=1.0, key="volume_ratio_input")
        trend_score = st.number_input("Trend Score", value=0.0, key="trend_score_input")
        volatility = st.number_input("Volatility", value=0.0, key="volatility_input")
        indicators = {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "price": price,
            "volume_ratio": volume_ratio,
            "trend_score": trend_score,
            "volatility": volatility
        }
        score = ml_filter.predict_signal_quality(indicators)
        st.metric("Predicted ML Score", f"{score:.2f}")
        st.markdown("""
        ## 🧠 How ML Works with Signals & Trades
        1. **Signals Collection**:  
           - Signals are generated by your strategies with indicators like RSI, MACD, Bollinger Bands, volatility, etc.
        2. **ML Filtering**:  
           - ML models score signals based on historical patterns. Only signals above a threshold are considered for trading.
        3. **Training ML Model**:  
           - Uses trades or signals as historical data, with indicators as features and profit/score as target.
           - Feature importance shows which indicators influence the ML prediction the most.
        4. **Generating Trades**:  
           - Trades can be created from filtered signals, inheriting key parameters like side, entry price, leverage.
        5. **Live Scoring**:  
           - Input current market indicators to get instant ML probability scores for new signals.
        """)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#888;'> AlgoTrader Pro - Multi Exchange (Binance & Bybit) v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exchange: {EXCHANGE.capitalize()}</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()