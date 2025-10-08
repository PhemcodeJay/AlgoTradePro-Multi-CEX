import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
from db import DatabaseManager, Signal
from signal_generator import analyze_single_symbol, get_signal_summary, generate_signals
from ml import MLFilter
from logging_config import get_trading_logger
from notifications import send_all_notifications

# Page configuration
st.set_page_config(
    page_title="Signals - AlgoTraderPro V2.0",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for white background
st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    st.subheader("Generate Signal")
    symbol_input = st.text_input("Symbol", value="BTCUSDT", placeholder="Enter symbol (e.g., BTCUSDT)")

with col2:
    interval = st.selectbox("Timeframe", ["60", "240", "1440"], index=0,
                           format_func=lambda x: {"60": "1 Hour", "240": "4 Hours", "1440": "1 Day"}[x])

with col3:
    st.write("")
    top_n = st.number_input("Number of Signals", min_value=1, max_value=50, value=10)

# Action buttons
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Analyze Symbol", type="primary", width="stretch"):
        if symbol_input:
            with st.spinner(f"Analyzing {symbol_input}..."):
                try:
                    result = analyze_single_symbol(current_exchange, symbol_input.upper(), interval)
                    if isinstance(result, dict) and result.get('symbol'):
                        signal = Signal(
                            symbol=result['symbol'],
                            interval=interval,
                            signal_type=result['signal_type'],
                            side=result['side'],
                            score=result['score'],
                            entry=result['entry'],
                            sl=result['sl'],
                            tp=result['tp'],
                            trail=0.0,
                            liquidation=0.0,
                            leverage=result['leverage'],
                            margin_usdt=0.0,
                            market=result['signal_type'].split('_')[0],  # e.g., 'mean_reversion' or 'trend_following'
                            indicators=result['indicators'],
                            exchange=current_exchange
                        )
                        if db_manager.add_signal(signal):
                            st.success(f"✅ Signal generated and saved for {symbol_input}")
                            st.rerun()
                        else:
                            st.warning(f"⚠️ Failed to save signal for {symbol_input}")
                    else:
                        st.warning(f"⚠️ No signal generated for {symbol_input}. This may be due to API restrictions or insufficient data.")
                except Exception as e:
                    error_msg = str(e)
                    if "451" in error_msg or "403" in error_msg or "restricted location" in error_msg:
                        st.error(f"⚠️ API access restricted for {symbol_input}. Try using a VPN or deploying in an unrestricted region.")
                    else:
                        st.error(f"Error analyzing {symbol_input}: {e}")
                    logger.error(f"Analyze symbol error: {e}")

with col2:
    if st.button("🚀 Generate Signals", type="primary", width="stretch"):
        with st.spinner(f"Generating top {top_n} signals..."):
            try:
                signals = generate_signals(
                    exchange=current_exchange,
                    timeframe=interval,
                    max_symbols=top_n
                )
                saved_signals = 0
                for result in signals:
                    if result.get('side') != "HOLD":
                        signal = Signal(
                            symbol=result['symbol'],
                            interval=interval,
                            signal_type=result['signal_type'],
                            side=result['side'],
                            score=result['score'],
                            entry=result['entry'],
                            sl=result['sl'],
                            tp=result['tp'],
                            trail=0.0,
                            liquidation=0.0,
                            leverage=result['leverage'],
                            margin_usdt=0.0,
                            market=result['signal_type'].split('_')[0],
                            indicators=result['indicators'],
                            exchange=current_exchange
                        )
                        if db_manager.add_signal(signal):
                            saved_signals += 1
                if saved_signals > 0:
                    st.success(f"✅ Generated and saved {saved_signals} signals successfully!")
                    st.rerun()
                else:
                    st.warning(f"No valid signals generated. This may be due to API restrictions or insufficient data.")
            except Exception as e:
                error_str = str(e)
                if "403" in error_str or "451" in error_str or "restricted location" in error_str:
                    st.error(f"⚠️ API Access Restricted for {current_exchange.title()}")
                    st.info(
                        f"The {current_exchange.title()} API rejected the request. "
                        "Try using a VPN or deploying in an unrestricted region (e.g., Europe via AWS). "
                        "Alternatively, switch to another exchange in the settings."
                    )
                else:
                    st.error(f"Error generating signals: {e}")
                logger.error(f"Signal generation error: {e}")

with col3:
    if st.button("📢 Send Notifications", width="stretch"):
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
            logger.error(f"Notification error: {e}")

with col4:
    if st.button("🔄 Refresh", width="stretch"):
        st.rerun()

st.divider()

# Signal statistics
st.subheader("📊 Signal Statistics")

try:
    recent_signals = db_manager.get_signals(limit=100, exchange=current_exchange)
    if recent_signals:
        signal_dicts = [s.to_dict() for s in recent_signals]
        summary = get_signal_summary(signal_dicts)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Signals", summary.get('total', 0))
        with col2:
            avg_score = summary.get('avg_score', 0.0)
            st.metric("Average Score", f"{avg_score:.1f}")
        with col3:
            st.metric("Top Symbol", summary.get('top_symbol', 'N/A'))

        col1, col2 = st.columns(2)
        with col1:
            side_counts = {'BUY': 0, 'SELL': 0}
            for s in signal_dicts:
                side = s.get('side', '').upper()
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
            st.plotly_chart(fig_pie, width="stretch")

        with col2:
            market_counts = {}
            for s in signal_dicts:
                market = s.get('signal_type', 'Unknown').split('_')[0]
                market_counts[market] = market_counts.get(market, 0) + 1
            fig_bar = px.bar(
                x=list(market_counts.keys()),
                y=list(market_counts.values()),
                title="Signal Types",
                labels={'x': 'Signal Type', 'y': 'Count'}
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("No recent signals available")
except Exception as e:
    st.error(f"Error loading signal statistics: {e}")
    logger.error(f"Statistics error: {e}")

st.divider()

st.subheader("📋 Recent Signals")

# Filters
col1, col2 = st.columns(2)
with col1:
    min_score = st.number_input("Min Score", min_value=0.0, max_value=100.0, value=0.0)
with col2:
    side_filter = st.selectbox("Side", ["All", "Buy", "Sell"])

filtered_signals = [s for s in recent_signals if (getattr(s, "score", 0) or 0) >= min_score]
if side_filter != "All":
    buy_terms = ['BUY', 'LONG'] if side_filter == "Buy" else ['SELL', 'SHORT']
    filtered_signals = [s for s in filtered_signals if (getattr(s, "side", "") or "").upper() in buy_terms]

if filtered_signals:
    signal_data = []
    for signal in filtered_signals[:50]:
        created_at = getattr(signal, "created_at", None)
        created_str = created_at.strftime('%Y-%m-%d %H:%M') if isinstance(created_at, datetime) else 'N/A'
        score_val = getattr(signal, "score", 0.0) or 0.0
        entry_val = getattr(signal, "entry", 0.0) or 0.0
        signal_data.append({
            'Symbol': getattr(signal, "symbol", "N/A"),
            'Side': getattr(signal, "side", "N/A"),
            'Score': f"{score_val:.1f}",
            'Entry': f"${entry_val:.6f}",
            'Signal Type': getattr(signal, "signal_type", "N/A"),
            'Created': created_str
        })

    df = pd.DataFrame(signal_data)
    def color_side(val):
        if val.upper() in ['BUY', 'LONG']:
            return 'color: green'
        elif val.upper() in ['SELL', 'SHORT']:
            return 'color: red'
        return ''
    styled_df = df.style.map(color_side, subset=['Side'])
    st.dataframe(styled_df, width="stretch", height=300)
else:
    st.info("No signals found")

# Detailed signal view
st.divider()
st.subheader("🔍 Signal Details")

try:
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
                st.write(f"Score: {getattr(selected_signal, 'score', 0.0):.1f}")
                st.write(f"Signal Type: {getattr(selected_signal, 'signal_type', 'N/A')}")
                st.write(f"Entry Price: ${getattr(selected_signal, 'entry', 0.0):.6f}")
                st.write(f"Take Profit: ${getattr(selected_signal, 'tp', 0.0):.6f}")
                st.write(f"Stop Loss: ${getattr(selected_signal, 'sl', 0.0):.6f}")
                st.write(f"Leverage: {getattr(selected_signal, 'leverage', 0)}x")

            with col2:
                st.write("**Technical Indicators:**")
                indicators = getattr(selected_signal, "indicators", {})
                if isinstance(indicators, dict):
                    for key, value in indicators.items():
                        if isinstance(value, (int, float)):
                            st.write(f"{key.replace('_', ' ').title()}: {value:.4f}")
                        else:
                            st.write(f"{key.replace('_', ' ').title()}: {value}")
                else:
                    st.write("No indicator data available")

            st.divider()
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📈 Execute Virtual Trade", type="primary", width="stretch"):
                    try:
                        if not trading_engine or not trading_engine.client:
                            st.error("Trading engine not properly initialized. Cannot execute trade.")
                        else:
                            success = trading_engine.execute_virtual_trade(selected_signal.to_dict())
                            if success:
                                st.success(f"Virtual trade executed for {getattr(selected_signal, 'symbol', 'N/A')}")
                            else:
                                st.error("Failed to execute virtual trade")
                    except Exception as e:
                        st.error(f"Error executing trade: {e}")
                        logger.error(f"Trade execution error: {e}")

            with col2:
                if st.button("🔍 ML Analysis", width="stretch"):
                    try:
                        ml_filter = MLFilter()
                        indicators = getattr(selected_signal, "indicators", {})
                        if isinstance(indicators, dict):
                            features = ml_filter.prepare_features(indicators)
                            if features is not None and getattr(features, "size", 0) > 0 and ml_filter.model is not None:
                                scaled = ml_filter.scaler.transform(features)
                                prob = ml_filter.model.predict_proba(scaled)[0][1]
                                st.info(f"ML Quality Score: {prob:.3f} ({prob*100:.1f}%)")
                                importance = ml_filter.get_feature_importance()
                                if importance:
                                    st.write("**Feature Importance:**")
                                    if isinstance(importance, dict):
                                        for feature, imp in list(importance.items())[:5]:
                                            st.write(f"{feature}: {imp:.3f}")
                                    else:
                                        st.write(str(importance))
                            else:
                                st.warning("No ML model available or insufficient features")
                        else:
                            st.warning("No valid indicator data for ML analysis")
                    except Exception as e:
                        st.error(f"ML analysis failed: {e}")
                        logger.error(f"ML analysis error: {e}")
except Exception as e:
    st.error(f"Error loading signal details: {e}")
    logger.error(f"Signal details error: {e}")

# Signal trends
st.divider()
st.subheader("📈 Signal Trends")

try:
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    raw_signals = db_manager.get_signals(limit=500, exchange=current_exchange)
    recent_signals = [s for s in raw_signals if isinstance(getattr(s, "created_at", None), datetime) and getattr(s, "created_at") > week_ago]

    if recent_signals:
        daily_counts = {}
        daily_avg_scores = {}
        for signal in recent_signals:
            created_at = getattr(signal, "created_at", None)
            if not isinstance(created_at, datetime):
                continue
            date_key = created_at.date()
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
            daily_avg_scores[date_key] = daily_avg_scores.get(date_key, []) + [getattr(signal, "score", 0.0) or 0.0]

        for date_key in daily_avg_scores:
            scores = daily_avg_scores[date_key]
            daily_avg_scores[date_key] = sum(scores) / len(scores) if scores else 0

        dates = sorted(daily_counts.keys())
        counts = [daily_counts[d] for d in dates]
        avg_scores = [daily_avg_scores[d] for d in dates]

        col1, col2 = st.columns(2)
        with col1:
            fig_counts = px.line(x=dates, y=counts, title="Daily Signal Count")
            fig_counts.update_layout(height=300)
            st.plotly_chart(fig_counts, width="stretch")
        with col2:
            fig_scores = px.line(x=dates, y=avg_scores, title="Average Signal Score")
            fig_scores.update_layout(height=300)
            st.plotly_chart(fig_scores, width="stretch")
    else:
        st.info("No recent signals for trend analysis")
except Exception as e:
    st.error(f"Error loading signal trends: {e}")
    logger.error(f"Trends error: {e}")