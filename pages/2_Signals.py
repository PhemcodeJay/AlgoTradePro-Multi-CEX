import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from db import DatabaseManager
from signal_generator import analyze_single_symbol, get_signal_summary, generate_signals, get_symbols
from ml import MLFilter
from logging_config import get_trading_logger
from notifications import send_all_notifications

# Page configuration
st.set_page_config(
    page_title="Signals - AlgoTrader Pro",
    page_icon="📡",
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

# Initialize logger
logger = get_trading_logger(__name__)

st.title("📡 Trading Signals")

# Signal controls
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    # Manual signal generation
    st.subheader("Generate Signal")
    symbol_input = st.text_input("Symbol", value="BTCUSDT", placeholder="Enter symbol (e.g., BTCUSDT)")

with col2:
    interval = st.selectbox("Timeframe", ["60", "240", "1440"], index=0, 
                          format_func=lambda x: {"60": "1 Hour", "240": "4 Hours", "1440": "1 Day"}[x])

with col3:
    st.write("")  # Spacing
    top_n = st.number_input("Number of Signals", min_value=1, max_value=50, value=10)

# Action buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Analyze Symbol", type="primary", use_container_width=True):
        if symbol_input:
            with st.spinner(f"Analyzing {symbol_input}..."):
                try:
                    result = analyze_single_symbol(symbol_input.upper(), interval)
                    # be defensive: ensure result is a dict-like with a real symbol string
                    if isinstance(result, dict) and bool(result.get('symbol')):
                        st.success(f"✅ Signal generated for {symbol_input}")
                        st.rerun()
                    else:
                        st.warning(f"⚠️ No signal generated for {symbol_input}. This may be due to API restrictions or insufficient data.")
                except Exception as e:
                    error_msg = str(e)
                    if "451" in error_msg or "restricted location" in error_msg:
                        st.error(f"⚠️ API access restricted for {symbol_input}. Try using Bybit exchange or check your VPN settings.")
                    else:
                        st.error(f"Error analyzing {symbol_input}: {e}")

with col2:
    if st.button("🚀 Generate Signals", type="primary", use_container_width=True):
        with st.spinner(f"Generating top {top_n} signals..."):
            try:
                from settings import load_settings
                
                settings = load_settings()
                
                # Get symbols from settings or use fallback
                symbols = settings.get("SYMBOLS", ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"])
                
                # Try to get live symbols if API is available
                try:
                    live_symbols = get_symbols(limit=100)
                    if live_symbols:
                        symbols = live_symbols[:100]
                except Exception:
                    logger.warning("Using fallback symbols due to API restrictions")
                
                signals = generate_signals(
                    symbols=symbols[:50],  # Limit to avoid timeout
                    interval=interval,
                    top_n=top_n
                )
                
                if signals:
                    st.success(f"✅ Generated {len(signals)} signals successfully!")
                    st.rerun()
                else:
                    st.warning("⚠️ No signals generated. This may be due to API restrictions or market conditions.")
            except Exception as e:
                st.error(f"Error generating signals: {e}")
                logger.error(f"Signal generation error: {e}")

with col3:
    if st.button("📢 Send Notifications", use_container_width=True):
        try:
            recent_signals = db_manager.get_signals(limit=10, exchange=current_exchange)
            if recent_signals:
                signal_dicts = [s.to_dict() for s in recent_signals]
                send_all_notifications(signal_dicts)
                st.success(f"Notifications sent for {len(signal_dicts)} signals!")
            else:
                st.warning("No signals available to send")
        except Exception as e:
            st.error(f"Error sending notifications: {e}")

with col4:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.divider()

# Signal statistics
st.subheader("📊 Signal Statistics")

try:
    # Get recent signals for statistics
    recent_signals = db_manager.get_signals(limit=100, exchange=current_exchange)
    
    if recent_signals:
        # Convert to dict format for summary
        signal_dicts = []
        for signal in recent_signals:
            # to_dict() already returns safe python types (per your db model), so use it
            signal_dict = signal.to_dict()
            signal_dicts.append(signal_dict)
        
        summary = get_signal_summary(signal_dicts)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Signals", summary.get('total', 0))
        
        with col2:
            avg_score = summary.get('avg_score', 0.0)
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col3:
            st.metric("Top Symbol", summary.get('top_symbol', 'N/A'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution pie chart
            side_counts = {'BUY': 0, 'SELL': 0}
            for s in signal_dicts:
                side = (s.get('side') or '').upper()
                if side in ['BUY', 'LONG']:
                    side_counts['BUY'] += 1
                elif side in ['SELL', 'SHORT']:
                    side_counts['SELL'] += 1
            
            fig_pie = px.pie(
                values=list(side_counts.values()),
                names=list(side_counts.keys()),
                title="Signal Distribution",
                hole=0.3
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Market types bar chart
            market_counts = {}
            for s in signal_dicts:
                market = s.get('market') or 'Unknown'
                market_counts[market] = market_counts.get(market, 0) + 1
            
            fig_bar = px.bar(
                x=list(market_counts.keys()),
                y=list(market_counts.values()),
                title="Market Types",
                labels={'x': 'Market', 'y': 'Count'}
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("No recent signals available")
        
except Exception as e:
    st.error(f"Error loading signal statistics: {e}")

st.divider()

st.subheader("📋 Recent Signals")

# Filters
col1, col2 = st.columns(2)

with col1:
    min_score = st.number_input("Min Score", min_value=0.0, max_value=100.0, value=0.0)

with col2:
    side_filter = st.selectbox("Side", ["All", "Buy", "Sell"])

# defensive: ensure recent_signals variable exists
recent_signals = recent_signals if 'recent_signals' in locals() else db_manager.get_signals(limit=100, exchange=current_exchange)

filtered_signals = [s for s in recent_signals if (getattr(s, "score", 0) or 0) >= min_score]

if side_filter != "All":
    buy_terms = ['BUY', 'LONG'] if side_filter == "Buy" else ['SELL', 'SHORT']
    # use getattr and upper safely
    filtered_signals = [s for s in filtered_signals if (getattr(s, "side", "") or "").upper() in buy_terms]

if filtered_signals:
    signal_data = []
    for signal in filtered_signals[:50]:
        created_at = getattr(signal, "created_at", None)
        created_str = created_at.strftime('%Y-%m-%d %H:%M') if isinstance(created_at, datetime) else 'N/A'
        # Score and entry should be numbers on model; be defensive with getattr
        score_val = getattr(signal, "score", 0.0) or 0.0
        entry_val = getattr(signal, "entry", 0.0) or 0.0

        signal_data.append({
            'Symbol': getattr(signal, "symbol", "N/A"),
            'Side': getattr(signal, "side", "N/A"),
            'Score': f"{score_val:.1f}",
            'Entry': f"${entry_val:.6f}",
            'Market': getattr(signal, "market", "N/A"),
            'Created': created_str
        })
    
    df = pd.DataFrame(signal_data)
    
    # Color code by side
    def color_side(val):
        if val.upper() in ['BUY', 'LONG']:
            return 'color: green'
        else:
            return 'color: red'
    
    styled_df = df.style.map(color_side, subset=['Side'])
    st.dataframe(styled_df, use_container_width=True, height=300)
    
else:
    st.info("No signals found")

# Detailed signal view
st.divider()
st.subheader("🔍 Signal Details")

try:
    # Build keys safely, ensuring created_at is datetime
    signal_keys = []
    for s in filtered_signals:
        created_at = getattr(s, "created_at", None)
        created_time = created_at.strftime('%H:%M:%S') if isinstance(created_at, datetime) else "N/A"
        signal_keys.append(f"{getattr(s, 'symbol', 'N/A')} - {created_time}")
    
    selected_key = st.selectbox("Select signal for details", ["None"] + signal_keys)
    
    if selected_key != "None":
        selected_signal = None
        for signal in filtered_signals:
            created_at = getattr(signal, "created_at", None)
            created_time = created_at.strftime('%H:%M:%S') if isinstance(created_at, datetime) else "N/A"
            if f"{getattr(signal, 'symbol', 'N/A')} - {created_time}" == selected_key:
                selected_signal = signal
                break
        
        if selected_signal:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                st.write(f"Symbol: {getattr(selected_signal, 'symbol', 'N/A')}")
                st.write(f"Side: {getattr(selected_signal, 'side', 'N/A')}")
                st.write(f"Score: {getattr(selected_signal, 'score', 0.0):.1f}%")
                st.write(f"Market Type: {getattr(selected_signal, 'market', 'N/A')}")
                st.write(f"Entry Price: ${getattr(selected_signal, 'entry', 0.0):.6f}")
                st.write(f"Take Profit: ${getattr(selected_signal, 'tp', 0.0):.6f}")
                st.write(f"Stop Loss: ${getattr(selected_signal, 'sl', 0.0):.6f}")
                st.write(f"Leverage: {getattr(selected_signal, 'leverage', 0)}x")
            
            with col2:
                st.write("**Technical Indicators:**")
                indicators_raw = getattr(selected_signal, "indicators", None)
                # Ensure indicators is a real dict before iterating / sending to ML
                indicators = indicators_raw if isinstance(indicators_raw, dict) else {}
                
                if indicators:
                    for key, value in indicators.items():
                        if isinstance(value, (int, float)):
                            st.write(f"{key.upper()}: {value:.4f}")
                        else:
                            st.write(f"{key.upper()}: {value}")
                else:
                    st.write("No indicator data available")
            
            # Execute trade button
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("📈 Execute Virtual Trade", type="primary"):
                    try:
                        signal_dict = selected_signal.to_dict()
                        success = trading_engine.execute_virtual_trade(signal_dict)
                        if success:
                            st.success(f"Virtual trade executed for {getattr(selected_signal, 'symbol', 'N/A')}")
                        else:
                            st.error("Failed to execute virtual trade")
                    except Exception as e:
                        st.error(f"Error executing trade: {e}")
            
            with col2:
                if st.button("🔍 ML Analysis"):
                    try:
                        ml_filter = MLFilter()
                        indicators_raw = getattr(selected_signal, "indicators", None)
                        indicators = indicators_raw if isinstance(indicators_raw, dict) else {}
                        # prepare_features expects a Dict[str, Any] — we now pass a dict
                        features = ml_filter.prepare_features(indicators)
                        # defensive checks on features and model
                        if features is not None and getattr(features, "size", 0) > 0 and ml_filter.model is not None:
                            scaled = ml_filter.scaler.transform(features)
                            prob = ml_filter.model.predict_proba(scaled)[0][1]
                            st.info(f"ML Quality Score: {prob:.3f} ({prob*100:.1f}%)")
                            
                            # Show feature importance
                            importance = ml_filter.get_feature_importance()
                            if importance:
                                st.write("**Feature Importance:**")
                                # if importance is dict-like
                                if isinstance(importance, dict):
                                    for feature, imp in list(importance.items())[:5]:
                                        st.write(f"{feature}: {imp:.3f}")
                                else:
                                    st.write(str(importance))
                        else:
                            st.warning("No ML model available or insufficient features")
                    except Exception as e:
                        st.error(f"ML analysis failed: {e}")
    
except Exception as e:
    st.error(f"Error loading signals: {e}")

# Signal generation trends
st.divider()
st.subheader("📈 Signal Trends")

try:
    # Get signals from last 7 days
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    raw_signals = db_manager.get_signals(limit=500, exchange=current_exchange)
    # Only include signals with a real datetime created_at and which are after week_ago
    recent_signals = [s for s in raw_signals if isinstance(getattr(s, "created_at", None), datetime) and getattr(s, "created_at") > week_ago]
    
    if recent_signals:
        # Group by date
        daily_counts = {}
        daily_avg_scores = {}
        
        for signal in recent_signals:
            created_at = getattr(signal, "created_at", None)
            if not isinstance(created_at, datetime):
                continue
            date_key = created_at.date()
            if date_key not in daily_counts:
                daily_counts[date_key] = 0
                daily_avg_scores[date_key] = []
            
            daily_counts[date_key] += 1
            daily_avg_scores[date_key].append(getattr(signal, "score", 0.0) or 0.0)
        
        # Calculate average scores
        for date_key in daily_avg_scores:
            scores = daily_avg_scores[date_key]
            daily_avg_scores[date_key] = sum(scores) / len(scores) if scores else 0
        
        # Create charts
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[d] for d in dates]
        avg_scores = [daily_avg_scores[d] for d in dates]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_counts = px.line(x=dates, y=counts, title="Daily Signal Count")
            fig_counts.update_layout(height=300)
            st.plotly_chart(fig_counts, use_container_width=True)
        
        with col2:
            fig_scores = px.line(x=dates, y=avg_scores, title="Average Signal Score")
            fig_scores.update_layout(height=300)
            st.plotly_chart(fig_scores, use_container_width=True)
    
    else:
        st.info("No recent signals for trend analysis")

except Exception as e:
    st.error(f"Error loading signal trends: {e}")
