import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from db import db_manager, Signal
from signal_generator import analyze_single_symbol, get_signal_summary, generate_signals, convert_np_types
from ml import MLFilter
from logging_config import get_trading_logger
from notifications import send_all_notifications
import numpy as np
import bcrypt

# Page configuration
st.set_page_config(
    page_title="Signals - AlgoTraderPro V2.0",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Header
st.markdown("""
<div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0;'>📡 Trading Signals</h1>
    <p style='color: white; margin: 0.5rem 0 0 0;'>Generate and analyze trading signals for {}</p>
</div>
""".format(st.session_state.get('user', {}).get('username', 'N/A')), unsafe_allow_html=True)



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

# Check authentication
if not st.session_state.get('authenticated', False):
    st.markdown("### 🔐 Signals Access")
    st.markdown("""
    **Instructions:**
    - **Login**: Enter your username and password to access trading signals.
    - **Register**: Create a new account if you don't have one. Passwords must be at least 6 characters.
    - **Note**: Signals are generated based on your selected exchange and trading mode.
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
if 'trading_engine' not in st.session_state or 'current_exchange' not in st.session_state:
    st.error("Trading engine not initialized. Please restart the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
current_exchange = st.session_state.current_exchange
user_id = st.session_state.get('user', {}).get('id', None)
account_type = st.session_state.get('account_type', 'virtual')
username = st.session_state.get('user', {}).get('username', 'N/A')

# Quick Info Cards
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

st.divider()

st.markdown("""
### 📡 Signal Generation
**Instructions:**
- Enter a symbol (e.g., BTCUSDT) to analyze and generate a trading signal.
- Select a timeframe for analysis (1 Hour, 4 Hours, 1 Day).
- Specify the number of top signals to generate.
- Use filters to view recent signals by score or side.
""")

col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    st.markdown("**Generate Signal**")
    symbol_input = st.text_input("Symbol", value="BTCUSDT", placeholder="Enter symbol (e.g., BTCUSDT)")

with col2:
    interval = st.selectbox("Timeframe", ["60", "240", "1440"], index=0,
                           format_func=lambda x: {"60": "1 Hour", "240": "4 Hours", "1440": "1 Day"}[x])

with col3:
    st.write("")
    top_n = st.number_input("Number of Signals", min_value=1, max_value=50, value=10)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔍 Analyze Symbol", type="primary", use_container_width=True):
        if symbol_input:
            with st.spinner(f"Analyzing {symbol_input}..."):
                try:
                    result = analyze_single_symbol(current_exchange, symbol_input.upper(), interval)
                    if isinstance(result, dict) and result.get('symbol'):
                        signal = {
                            'symbol': str(result.get('symbol', '')),
                            'interval': str(interval),
                            'signal_type': str(result.get('signal_type', 'neutral')),
                            'side': str(result.get('side', 'HOLD')),
                            'score': float(result.get('score', 0.0)),
                            'entry': float(result.get('entry', 0.0)),
                            'sl': float(result.get('sl', 0.0)),
                            'tp': float(result.get('tp', 0.0)),
                            'trail': float(result.get('trail', 0.0)),
                            'liquidation': float(result.get('liquidation', 0.0)),
                            'leverage': int(result.get('leverage', 1)),
                            'margin_usdt': float(result.get('margin_usdt', 0.0)),
                            'market': str(result.get('signal_type', 'neutral').split('_')[0]),
                            'indicators': convert_np_types(result.get('indicators', {})),
                            'exchange': str(current_exchange),
                            'created_at': str(result.get('created_at', datetime.now(timezone.utc).isoformat())),
                            'user_id': user_id
                        }
                        if not isinstance(signal['indicators'], dict):
                            logger.error(f"Invalid indicators for {symbol_input}: {signal['indicators']}")
                            st.warning(f"⚠️ Invalid indicator data for {symbol_input}")
                        elif db_manager.add_signal(signal):
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
        else:
            st.error("Please enter a valid symbol")

with col2:
    if st.button("🚀 Generate Top Signals", type="primary", use_container_width=True):
        with st.spinner(f"Generating top {top_n} signals..."):
            try:
                signals = generate_signals(
                    exchange=current_exchange,
                    timeframe=interval,
                    max_symbols=top_n,
                    user_id=user_id
                )
                saved = 0
                for signal in signals:
                    signal['interval'] = str(interval)
                    signal['exchange'] = str(current_exchange)
                    signal['created_at'] = str(datetime.now(timezone.utc).isoformat())
                    signal['user_id'] = user_id
                    signal['indicators'] = convert_np_types(signal.get('indicators', {}))
                    if db_manager.add_signal(signal):
                        saved += 1
                if saved > 0:
                    st.success(f"✅ Generated and saved {saved} signals")
                    st.rerun()
                else:
                    st.warning("⚠️ No new signals generated or saved")
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
    if st.button("📢 Send Notifications", type="primary", use_container_width=True):
        with st.spinner("Sending notifications..."):
            try:
                signals = db_manager.get_signals(limit=top_n, exchange=current_exchange, user_id=user_id)
                if signals:
                    sent = send_all_notifications(signals)
                    st.success(f"✅ Sent notifications for {sent} signals")
                else:
                    st.info("📭 No signals available to notify")
            except Exception as e:
                logger.error(f"Error sending notifications: {e}")
                st.error(f"Error sending notifications: {e}")

with col4:
    if st.button("🔄 Refresh", type="primary", use_container_width=True):
        st.rerun()

st.divider()

st.markdown("""
### 📊 Signal Overview
**Instructions:**
- Filter signals by score or side to analyze specific trading opportunities.
- View detailed signal data in the table below.
- Use the charts to visualize signal scores and side distribution.
""")

col1, col2 = st.columns(2)
with col1:
    score_filter = st.slider("Minimum Signal Score", 0.0, 1.0, 0.0, step=0.05)
with col2:
    side_filter = st.multiselect("Signal Side", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL"])

signals = db_manager.get_signals(limit=100, exchange=current_exchange, user_id=user_id)
filtered_signals = [
    s for s in signals
    if s.get('score', 0.0) >= score_filter and (not side_filter or s.get('side', 'HOLD') in side_filter)
]

if filtered_signals:
    signal_data = []
    for signal in filtered_signals:
        created_at = datetime.fromisoformat(signal['created_at']) if signal.get('created_at') else None
        signal_data.append({
            'Symbol': signal.get('symbol', 'N/A'),
            'Side': signal.get('side', 'HOLD'),
            'Score': f"{signal.get('score', 0.0):.2f}",
            'Entry': f"${signal.get('entry', 0.0):.2f}",
            'SL': f"${signal.get('sl', 0.0):.2f}",
            'TP': f"${signal.get('tp', 0.0):.2f}",
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

    col1, col2 = st.columns(2)
    with col1:
        fig_scores = px.histogram(
            df_signals,
            x='Score',
            title="Signal Score Distribution",
            nbins=20,
            color_discrete_sequence=['#636EFA']
        )
        fig_scores.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_scores, use_container_width=True)

    with col2:
        fig_sides = px.pie(
            df_signals,
            names='Side',
            title="Signal Side Distribution",
            color='Side',
            color_discrete_map={'BUY': '#2ecc71', 'SELL': '#ff4d4f', 'HOLD': '#b0b0b0'}
        )
        fig_sides.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_sides, use_container_width=True)
else:
    st.info("📭 No signals match the selected filters")

st.divider()

st.markdown("""
### 🔍 Signal Details
**Instructions:**
- Select a signal from the dropdown to view detailed information, including technical indicators.
- Execute a virtual trade or run an ML analysis on the selected signal.
""")

signals = db_manager.get_signals(limit=100, exchange=current_exchange, user_id=user_id)
signal_keys = [
    f"{s.get('symbol', 'N/A')} - {datetime.fromisoformat(s['created_at']).strftime('%H:%M:%S') if s.get('created_at') else 'N/A'}"
    for s in signals
]

selected_key = st.selectbox("Select signal for details", ["None"] + signal_keys)
if selected_key != "None":
    selected_signal = None
    for signal in signals:
        created_time = datetime.fromisoformat(signal['created_at']).strftime('%H:%M:%S') if signal.get('created_at') else 'N/A'
        if f"{signal.get('symbol', 'N/A')} - {created_time}" == selected_key:
            selected_signal = signal
            break

    if selected_signal:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Basic Information**")
            st.write(f"**Symbol**: {selected_signal.get('symbol', 'N/A')}")
            st.write(f"**Side**: {selected_signal.get('side', 'N/A')}")
            st.write(f"**Score**: {selected_signal.get('score', 0.0):.2f}")
            st.write(f"**Signal Type**: {selected_signal.get('signal_type', 'N/A')}")
            st.write(f"**Entry Price**: ${selected_signal.get('entry', 0.0):.2f}")
            st.write(f"**Take Profit**: ${selected_signal.get('tp', 0.0):.2f}")
            st.write(f"**Stop Loss**: ${selected_signal.get('sl', 0.0):.2f}")
            st.write(f"**Leverage**: {selected_signal.get('leverage', 0)}x")

        with col2:
            st.markdown("**Technical Indicators**")
            indicators = selected_signal.get('indicators', {})
            if isinstance(indicators, dict):
                for key, value in indicators.items():
                    if isinstance(value, (int, float)):
                        st.write(f"{key.replace('_', ' ').title()}: {value:.4f}")
                    else:
                        st.write(f"{key.replace('_', ' ').title()}: {value}")
            else:
                st.write("No indicator data available")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📈 Execute Virtual Trade", type="primary", use_container_width=True):
                try:
                    if not trading_engine or not trading_engine.client:
                        st.error("Trading engine not properly initialized. Cannot execute trade.")
                    else:
                        success = trading_engine.execute_virtual_trade(selected_signal)
                        if success:
                            st.success(f"Virtual trade executed for {selected_signal.get('symbol', 'N/A')}")
                        else:
                            st.error("Failed to execute virtual trade")
                except Exception as e:
                    st.error(f"Error executing trade: {e}")
                    logger.error(f"Trade execution error: {e}")

        with col2:
            if st.button("🔍 ML Analysis", type="primary", use_container_width=True):
                try:
                    ml_filter = MLFilter()
                    indicators = selected_signal.get('indicators', {})
                    if isinstance(indicators, dict):
                        features = ml_filter.prepare_features(indicators)
                        if features is not None and getattr(features, "size", 0) > 0 and ml_filter.model is not None:
                            scaled = ml_filter.scaler.transform(features)
                            prob = ml_filter.model.predict_proba(scaled)[0][1]
                            st.info(f"ML Quality Score: {prob:.3f} ({prob*100:.1f}%)")
                            importance = ml_filter.get_feature_importance()
                            if importance:
                                st.markdown("**Feature Importance**")
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

st.divider()

st.markdown("""
### 📈 Signal Trends
**Instructions:**
- View trends in signal counts and average scores over the past week.
- Analyze how signal generation has evolved to identify patterns.
""")

week_ago = datetime.now(timezone.utc) - timedelta(days=7)
signals = db_manager.get_signals(limit=500, exchange=current_exchange, user_id=user_id)
recent_signals = [
    s for s in signals
    if s.get('created_at') and datetime.fromisoformat(s['created_at'].replace('Z', '+00:00')) > week_ago
]

if recent_signals:
    daily_counts = {}
    daily_avg_scores = {}
    for signal in recent_signals:
        if not signal.get('created_at'):
            continue
        date_key = datetime.fromisoformat(signal['created_at'].replace('Z', '+00:00')).date()
        daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        daily_avg_scores[date_key] = daily_avg_scores.get(date_key, []) + [signal.get('score', 0.0)]

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
        st.plotly_chart(fig_counts, use_container_width=True)
    with col2:
        fig_scores = px.line(x=dates, y=avg_scores, title="Average Signal Score")
        fig_scores.update_layout(height=300)
        st.plotly_chart(fig_scores, use_container_width=True)
else:
    st.info("📭 No recent signals for trend analysis")

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