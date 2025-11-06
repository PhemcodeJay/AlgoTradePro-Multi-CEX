import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from db import db_manager, FeedbackModel, User, WalletBalance
from ml import MLFilter
from logging_config import get_trading_logger
from multi_trading_engine import TradingEngine
import bcrypt
import json
from datetime import datetime, timedelta, timezone

# Page configuration
st.set_page_config(
    page_title="AI Analysis - AlgoTraderPro V2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_exchange' not in st.session_state:
    st.session_state.current_exchange = os.getenv("EXCHANGE", "binance").lower()
if 'account_type' not in st.session_state:
    st.session_state.account_type = "virtual"
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = "N/A"

# Authentication functions
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
                exchange=st.session_state.current_exchange
            ))
            session.add(WalletBalance(
                user_id=new_user.id,
                account_type="real",
                available=0.0,
                used=0.0,
                total=0.0,
                currency="USDT",
                exchange=st.session_state.current_exchange
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
    st.markdown("### 🔐 AI Analysis Access")
    st.markdown("""
    **Instructions:**
    - **Login**: Enter your username and password to access AI analysis tools.
    - **Register**: Create a new account to manage ML models and feedback. Passwords must be at least 6 characters.
    - **Note**: ML settings are specific to your selected exchange and trading mode.
    """)
    
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.markdown("#### Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_login = st.form_submit_button("Login", type="primary", use_container_width=True)
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
            submit_register = st.form_submit_button("Register", type="primary", use_container_width=True)
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
    if st.button("Go to Main Page", type="primary", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

trading_engine: TradingEngine = st.session_state.trading_engine
current_exchange = st.session_state.current_exchange
account_type = st.session_state.account_type
user_id = st.session_state.user.get('id') if st.session_state.user else None
username = st.session_state.user.get('username', 'N/A')

if not user_id:
    st.error("User not authenticated. Please log in from the main page.")
    if st.button("Go to Main Page", type="primary", use_container_width=True):
        st.switch_page("app.py")
    st.stop()

# Header
st.markdown("""
<div style='padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0;'>🤖 AI Analysis & Training</h1>
    <p style='color: white; margin: 0.5rem 0 0 0;'>Manage machine learning models and feedback for {}</p>
</div>
""".format(username), unsafe_allow_html=True)

# ML Status Overview
st.markdown("### 📊 ML Status")
st.markdown("""
**Instructions:**
- Review the current status of your ML model and feedback data.
- Use this information to decide if model training or feedback adjustments are needed.
""")

try:
    ml_filter = MLFilter(user_id=st.session_state.user_id, exchange=current_exchange)
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_status = "✅ Loaded" if ml_filter.model is not None else "❌ Not Loaded"
        st.markdown(f"""
        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #667eea;'>
            <h4 style='margin: 0; color: #667eea;'>🤖 Model Status</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{model_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        feedback_count = len(db_manager.get_feedback(limit=1000, exchange=current_exchange, user_id=user_id))
        st.markdown(f"""
        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #764ba2;'>
            <h4 style='margin: 0; color: #764ba2;'>📈 Total Feedback</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{feedback_count}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recent_feedback = len(db_manager.get_feedback(limit=100, exchange=current_exchange, user_id=user_id))
        st.markdown(f"""
        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #10b981;'>
            <h4 style='margin: 0; color: #10b981;'>🏦 {current_exchange.title()} Feedback</h4>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{recent_feedback}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        recent_data = db_manager.get_feedback(limit=100, exchange=current_exchange, user_id=user_id)
        if recent_data:
            recent_data_list = [f.to_dict() for f in recent_data]
            positive_outcomes = len([f for f in recent_data_list if f['outcome']])
            accuracy = (positive_outcomes / len(recent_data_list)) * 100 if recent_data_list else 0
            st.markdown(f"""
            <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                <h4 style='margin: 0; color: #f59e0b;'>📊 Recent Accuracy</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{accuracy:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                <h4 style='margin: 0; color: #f59e0b;'>📊 Recent Accuracy</h4>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>N/A</p>
            </div>
            """, unsafe_allow_html=True)

except Exception as e:
    logger.error(f"Error loading ML status: {e}")
    st.error(f"Error loading ML status: {e}. Please check the MLFilter configuration.")
    st.stop()

st.divider()

# Helper functions
def get_feature_description(feature: str) -> str:
    descriptions = {
        'rsi': 'Relative Strength Index - momentum oscillator (0-100)',
        'macd': 'MACD line - trend following momentum indicator',
        'macd_signal': 'MACD signal line - smoothed MACD',
        'macd_histogram': 'MACD histogram - difference between MACD and signal',
        'bb_position': 'Bollinger Band position - price relative to bands (0-1)',
        'volume_ratio': 'Volume ratio - current vs average volume',
        'trend_score': 'Trend score - price momentum indicator',
        'volatility': 'Price volatility - price standard deviation',
        'price_change_1h': '1-hour price change percentage',
        'price_change_4h': '4-hour price change percentage',
        'price_change_24h': '24-hour price change percentage',
        'stoch_rsi_k': 'Stochastic RSI K line - momentum oscillator',
        'stoch_rsi_d': 'Stochastic RSI D line - smoothed Stochastic RSI',
        'sma_200': '200-period Simple Moving Average',
        'ema_9': '9-period Exponential Moving Average'
    }
    return descriptions.get(feature, 'Technical indicator')

def get_feature_performance(feature: str) -> str:
    try:
        trades = db_manager.get_trades(limit=100, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
        trades_list = [t.to_dict() for t in trades]
        valid_trades = [t for t in trades_list if t['indicators'] and t['pnl'] is not None]
        
        if not valid_trades:
            return "No data available"
        
        feature_values = []
        outcomes = []
        for trade in valid_trades:
            if feature in trade['indicators']:
                feature_values.append(trade['indicators'][feature])
                outcomes.append(1 if trade['pnl'] > 0 else 0)
        
        if not feature_values:
            return "Feature not found in trade data"
        
        df = pd.DataFrame({'feature': feature_values, 'outcome': outcomes})
        correlation = df['feature'].corr(df['outcome'])
        
        profitable = df[df['outcome'] == 1]['feature']
        unprofitable = df[df['outcome'] == 0]['feature']
        
        profitable_mean = profitable.mean() if len(profitable) > 0 else 0
        unprofitable_mean = unprofitable.mean() if len(unprofitable) > 0 else 0
        
        return f"Correlation: {correlation:.3f}, Profitable mean: {profitable_mean:.3f}, Unprofitable mean: {unprofitable_mean:.3f}"
    
    except Exception as e:
        logger.error(f"Error calculating feature performance for {feature}: {e}")
        return "Error calculating performance"

def get_feature_suggestion(feature: str) -> str:
    suggestions = {
        'rsi': 'Add RSI divergence signals and multi-timeframe RSI values',
        'macd': 'Combine with MACD crossover signals and histogram analysis',
        'bb_position': 'Include Bollinger Band squeeze detection',
        'volume_ratio': 'Add volume profile analysis',
        'volatility': 'Consider volatility breakout signals',
        'trend_score': 'Enhance with multi-timeframe trend alignment',
        'price_change_1h': 'Add momentum acceleration signals',
        'price_change_4h': 'Combine with support/resistance levels',
        'price_change_24h': 'Include market structure analysis',
        'stoch_rsi_k': 'Combine with Stochastic RSI crossovers',
        'stoch_rsi_d': 'Use with K-line for confirmation signals',
        'sma_200': 'Combine with other moving averages for trend confirmation',
        'ema_9': 'Use with longer EMAs for crossover signals'
    }
    return suggestions.get(feature, 'Consider additional technical indicators')

# ML Management Tabs
st.markdown("### ⚙️ Manage AI Analysis")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏋️ Model Training",
    "📈 Feedback Analysis",
    "🎯 Feature Importance",
    "📊 Model Performance",
    "✏️ Manual Feedback"
])

with tab1:
    st.markdown("""
    ### 🏋️ Model Training & Management
    **Instructions:**
    - Train the ML model with at least 50 valid trades.
    - Reset the model to clear existing training data.
    - Review training data statistics before training.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Training Data Overview**")
        try:
            trades = db_manager.get_trades(limit=1000, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
            trades_list = [t.to_dict() for t in trades]
            valid_trades = [t for t in trades_list if t['pnl'] is not None and t['indicators']]
            
            if valid_trades:
                profitable_trades = len([t for t in valid_trades if t['pnl'] > 0])
                
                training_stats = {
                    "Total Trades": len(valid_trades),
                    "Profitable Trades": profitable_trades,
                    "Unprofitable Trades": len(valid_trades) - profitable_trades,
                    "Success Rate": f"{(profitable_trades / len(valid_trades) * 100):.1f}%"
                }
                
                for key, value in training_stats.items():
                    st.markdown(f"**{key}:** {value}")
                
                indicator_coverage = (len([t for t in valid_trades if t['indicators']]) / len(valid_trades)) * 100
                
                st.progress(indicator_coverage / 100, text=f"Indicator Coverage: {indicator_coverage:.1f}%")
                
                if indicator_coverage < 80:
                    st.warning("⚠️ Low indicator coverage may reduce model quality")
                elif indicator_coverage >= 95:
                    st.success("✅ Excellent indicator coverage")
                
                recent_trades = [t for t in valid_trades 
                               if isinstance(t['created_at'], (datetime, str)) and 
                               (datetime.fromisoformat(t['created_at']) if isinstance(t['created_at'], str) else t['created_at']) > 
                               datetime.now(timezone.utc) - timedelta(days=30)]
                
                st.markdown(f"**Recent Trades (30 days):** {len(recent_trades)}")
                
            else:
                st.info("No valid trades available for training")
        
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            st.error(f"Error loading training data: {e}")
    
    with col2:
        st.markdown("**Model Actions**")
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            if len(valid_trades) >= 50:
                with st.spinner("Training ML model..."):
                    try:
                        training_data = [
                            {'indicators': t['indicators'], 'pnl': t['pnl']}
                            for t in valid_trades
                            if t['indicators'] and t['pnl'] is not None
                        ]
                        success = ml_filter.train_model(training_data)
                        if success:
                            st.success("✅ Model trained successfully!")
                            st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                            st.rerun()
                        else:
                            st.error("❌ Model training failed")
                    except Exception as e:
                        logger.error(f"Training error: {e}")
                        st.error(f"Training error: {e}")
            else:
                st.error(f"Need at least 50 trades for training. Current: {len(valid_trades)}")
        
        if ml_filter.model is not None:
            st.markdown("**Current Model**")
            st.markdown(f"- **Type:** Random Forest")
            st.markdown(f"- **Features:** {len(ml_filter.feature_columns)}")
            st.markdown(f"- **Exchange:** {current_exchange.title()}")
        
        if st.button("🗑️ Reset Model", type="secondary", use_container_width=True):
            if st.button("⚠️ Confirm Reset", key="confirm_model_reset", type="secondary", use_container_width=True):
                try:
                    if os.path.exists(ml_filter.model_path):
                        os.remove(ml_filter.model_path)
                    if os.path.exists(ml_filter.scaler_path):
                        os.remove(ml_filter.scaler_path)
                    ml_filter.model = None
                    ml_filter.scaler = None
                    st.success("✅ Model reset successfully!")
                    st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error resetting model: {e}")
                    st.error(f"Error resetting model: {e}")

with tab2:
    st.markdown("""
    ### 📈 Feedback Analysis
    **Instructions:**
    - Review feedback trends, symbol performance, and recent feedback.
    - Use charts and tables to identify patterns in model performance.
    """)
    
    try:
        feedback_data = db_manager.get_feedback(limit=500, exchange=current_exchange, user_id=user_id)
        
        if feedback_data:
            feedback_list = [f.to_dict() for f in feedback_data]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_count = len([f for f in feedback_list if f['outcome']])
                st.markdown(f"""
                <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #10b981;'>
                    <h4 style='margin: 0; color: #10b981;'>✅ Positive Outcomes</h4>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{positive_count}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                negative_count = len([f for f in feedback_list if not f['outcome']])
                st.markdown(f"""
                <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #ef4444;'>
                    <h4 style='margin: 0; color: #ef4444;'>❌ Negative Outcomes</h4>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{negative_count}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                accuracy = (positive_count / len(feedback_list)) * 100 if feedback_list else 0
                st.markdown(f"""
                <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                    <h4 style='margin: 0; color: #f59e0b;'>📊 Overall Accuracy</h4>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{accuracy:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**Feedback Trends**")
            feedback_df = pd.DataFrame([{
                'Date': (datetime.fromisoformat(f['timestamp']) if isinstance(f['timestamp'], str) else f['timestamp']).date(),
                'Outcome': f['outcome'],
                'Symbol': f['symbol'],
                'Exchange': f['exchange'],
                'Profit_Loss': f['profit_loss'] or 0
            } for f in feedback_list])
            
            daily_feedback = feedback_df.groupby('Date').agg({
                'Outcome': ['count', 'mean'],
                'Profit_Loss': 'sum'
            }).round(3)
            
            daily_feedback.columns = ['Total_Feedback', 'Success_Rate', 'Total_PnL']
            daily_feedback = daily_feedback.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_feedback_count = px.bar(
                    daily_feedback,
                    x='Date',
                    y='Total_Feedback',
                    title="Daily Feedback Count",
                    color='Total_Feedback',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_feedback_count, use_container_width=True)
            
            with col2:
                fig_success_rate = px.line(
                    daily_feedback,
                    x='Date',
                    y='Success_Rate',
                    title="Daily Success Rate",
                    markers=True
                )
                fig_success_rate.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                         annotation_text="Random Baseline (50%)")
                st.plotly_chart(fig_success_rate, use_container_width=True)
            
            st.markdown("**Performance by Symbol**")
            symbol_stats = feedback_df.groupby('Symbol').agg({
                'Outcome': ['count', 'mean'],
                'Profit_Loss': 'sum'
            }).round(3)
            
            symbol_stats.columns = ['Feedback_Count', 'Success_Rate', 'Total_PnL']
            symbol_stats = symbol_stats.reset_index()
            symbol_stats = symbol_stats.sort_values('Total_PnL', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Performers**")
                top_performers = symbol_stats.head(5)
                st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.markdown("**Bottom Performers**")
                bottom_performers = symbol_stats.tail(5)
                st.dataframe(bottom_performers, use_container_width=True)
            
            st.markdown("**Recent Feedback (Last 20)**")
            recent_feedback_data = []
            for feedback in feedback_list[:20]:
                timestamp = datetime.fromisoformat(feedback['timestamp']) if isinstance(feedback['timestamp'], str) else feedback['timestamp']
                recent_feedback_data.append({
                    'Date': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': feedback['symbol'],
                    'Outcome': '✅ Success' if feedback['outcome'] else '❌ Failure',
                    'P&L': f"${feedback['profit_loss']:.2f}" if feedback['profit_loss'] is not None else "N/A",
                    'Exchange': feedback['exchange']
                })
            
            df_recent = pd.DataFrame(recent_feedback_data)
            def color_outcome(val):
                if '✅' in str(val):
                    return 'background-color: rgba(0, 255, 0, 0.1)'
                elif '❌' in str(val):
                    return 'background-color: rgba(255, 0, 0, 0.1)'
                return ''
            
            styled_df = df_recent.style.applymap(color_outcome, subset=['Outcome'])
            st.dataframe(styled_df, use_container_width=True)
        
        else:
            st.info("No feedback data available")
    
    except Exception as e:
        logger.error(f"Error in feedback analysis: {e}")
        st.error(f"Error in feedback analysis: {e}")

with tab3:
    st.markdown("""
    ### 🎯 Feature Importance Analysis
    **Instructions:**
    - Review the importance of each feature in the ML model.
    - Analyze feature performance and get suggestions for improvement.
    - Ensure a trained model exists for accurate analysis.
    """)
    
    try:
        if ml_filter.model is not None:
            importance = ml_filter.get_feature_importance()
            if importance:
                features = list(importance.keys())
                importances = list(importance.values())
                
                fig_importance = px.bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    title="Feature Importance in ML Model",
                    color=importances,
                    color_continuous_scale='Viridis'
                )
                fig_importance.update_layout(height=500, xaxis_title="Importance")
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.markdown("**Feature Details**")
                importance_data = []
                for i, (feature, imp) in enumerate(importance.items()):
                    importance_data.append({
                        'Rank': i + 1,
                        'Feature': feature.upper(),
                        'Importance': f"{imp:.4f}",
                        'Percentage': f"{imp / sum(importances) * 100:.1f}%",
                        'Description': get_feature_description(feature),
                        'Performance': get_feature_performance(feature)
                    })
                
                df_importance = pd.DataFrame(importance_data)
                st.dataframe(df_importance, use_container_width=True)
                
                st.markdown("**Feature Analysis**")
                trades = db_manager.get_trades(limit=100, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
                trades_list = [t.to_dict() for t in trades]
                trades_with_indicators = [t for t in trades_list if t['indicators']]
                
                if len(trades_with_indicators) > 10:
                    feature_data = []
                    for trade in trades_with_indicators:
                        indicators = trade['indicators']
                        row = {'outcome': 1 if (trade['pnl'] or 0) > 0 else 0}
                        for feature in ml_filter.feature_columns:
                            row[feature] = indicators.get(feature, 0)
                        feature_data.append(row)
                    
                    df_features = pd.DataFrame(feature_data)
                    profitable_stats = df_features[df_features['outcome'] == 1].describe()
                    unprofitable_stats = df_features[df_features['outcome'] == 0].describe()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Profitable Trades Stats**")
                        st.dataframe(profitable_stats.round(4), height=300)
                    with col2:
                        st.markdown("**Unprofitable Trades Stats**")
                        st.dataframe(unprofitable_stats.round(4), height=300)
                
                st.markdown("**Feature Suggestions**")
                top_features = list(importance.keys())[:3]
                for feature in top_features:
                    st.markdown(f"- **{feature.upper()}:** {get_feature_suggestion(feature)}")
            
            else:
                st.warning("No feature importance data available")
        
        else:
            st.warning("No ML model loaded. Train a model first.")
    
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        st.error(f"Error in feature importance analysis: {e}")

with tab4:
    st.markdown("""
    ### 📊 Model Performance Metrics
    **Instructions:**
    - Evaluate model performance using accuracy, precision, recall, and F1-score.
    - Review ROC curve and prediction distribution for insights.
    - Ensure sufficient trade data for reliable metrics.
    """)
    
    try:
        if ml_filter.model is not None:
            st.markdown("**Model Validation**")
            trades = db_manager.get_trades(limit=200, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
            trades_list = [t.to_dict() for t in trades]
            valid_trades = [t for t in trades_list if t['indicators'] and t['pnl'] is not None]
            
            if len(valid_trades) > 20:
                X_val = []
                y_true = []
                for trade in valid_trades:
                    features = ml_filter.prepare_features(trade['indicators'])
                    if features.shape[1] > 0:
                        X_val.append(features.flatten())
                        y_true.append(1 if trade['pnl'] > 0 else 0)
                
                if len(X_val) > 10:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
                    
                    X_val = np.array(X_val)
                    y_pred = ml_filter.model.predict(ml_filter.scaler.transform(X_val))
                    y_proba = ml_filter.model.predict_proba(ml_filter.scaler.transform(X_val))[:, 1]
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"""
                        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #667eea;'>
                            <h4 style='margin: 0; color: #667eea;'>📏 Accuracy</h4>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{accuracy:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #764ba2;'>
                            <h4 style='margin: 0; color: #764ba2;'>🎯 Precision</h4>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{precision:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #10b981;'>
                            <h4 style='margin: 0; color: #10b981;'>🔄 Recall</h4>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{recall:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        st.markdown(f"""
                        <div style='padding: 1rem; background: #f0f2f6; border-radius: 8px; border-left: 4px solid #f59e0b;'>
                            <h4 style='margin: 0; color: #f59e0b;'>📊 F1-Score</h4>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem;'>{f1:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.3f})',
                        line=dict(color='blue', width=2)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_roc.update_layout(
                        title="ROC Curve",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=400
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                    st.markdown("**Prediction Distribution**")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pred_dist = px.histogram(
                            x=y_proba,
                            nbins=20,
                            title="Prediction Probability Distribution",
                            color_discrete_sequence=['blue']
                        )
                        fig_pred_dist.update_layout(xaxis_title="Predicted Probability", yaxis_title="Count")
                        st.plotly_chart(fig_pred_dist, use_container_width=True)
                    
                    with col2:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_true, y_pred)
                        fig_cm = px.imshow(
                            cm,
                            title="Confusion Matrix",
                            labels={'x': 'Predicted', 'y': 'Actual'},
                            x=['Negative', 'Positive'],
                            y=['Negative', 'Positive'],
                            color_continuous_scale='Blues'
                        )
                        fig_cm.update_layout(height=300)
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    st.markdown("**Accuracy by Confidence**")
                    confidence_data = [
                        {
                            'Confidence': prob,
                            'Actual': actual,
                            'Predicted': pred,
                            'Correct': actual == pred
                        }
                        for prob, actual, pred in zip(y_proba, y_true, y_pred)
                    ]
                    df_confidence = pd.DataFrame(confidence_data)
                    df_confidence['Confidence_Bin'] = pd.cut(df_confidence['Confidence'], bins=10)
                    confidence_stats = df_confidence.groupby('Confidence_Bin').agg({
                        'Correct': 'mean',
                        'Confidence': 'count'
                    }).reset_index()
                    fig_confidence = px.bar(
                        confidence_stats,
                        x='Confidence_Bin',
                        y='Correct',
                        title="Accuracy by Prediction Confidence",
                        color='Correct',
                        color_continuous_scale='Blues'
                    )
                    fig_confidence.update_layout(xaxis_title="Confidence Range", yaxis_title="Accuracy")
                    st.plotly_chart(fig_confidence, use_container_width=True)
                
                else:
                    st.warning("Insufficient validation data for detailed metrics")
            
            else:
                st.info("Need more valid trades for model validation")
        
        else:
            st.warning("No ML model loaded. Train a model first.")
    
    except Exception as e:
        logger.error(f"Error in model performance analysis: {e}")
        st.error(f"Error in model performance analysis: {e}")

with tab5:
    st.markdown("""
    ### ✏️ Manual Feedback Entry
    **Instructions:**
    - Add manual feedback to improve the ML model.
    - Specify the symbol, outcome, and technical indicators.
    - Ensure all inputs are valid to avoid errors.
    """)
    
    st.markdown("**Add Manual Feedback**")
    col1, col2 = st.columns(2)
    
    with col1:
        feedback_symbol = st.text_input("Symbol", value="BTCUSDT", placeholder="e.g., BTCUSDT")
        feedback_outcome = st.selectbox(
            "Trade Outcome",
            ["Success (Profitable)", "Failure (Loss)"],
            help="Was this trade profitable or not?"
        )
        profit_loss = st.number_input(
            "Profit/Loss Amount ($)",
            value=0.0,
            step=0.01,
            format="%.2f",
            help="Actual profit or loss amount"
        )
        st.markdown("**Signal Indicators**")
        rsi = st.slider("RSI", 0.0, 100.0, 50.0, 0.1, help="Relative Strength Index (0-100)")
        stoch_rsi_k = st.slider("Stoch RSI K", 0.0, 100.0, 50.0, 0.1, help="Stochastic RSI K line")
        stoch_rsi_d = st.slider("Stoch RSI D", 0.0, 100.0, 50.0, 0.1, help="Stochastic RSI D line")
        macd = st.number_input("MACD", value=0.0, step=0.001, format="%.4f", help="MACD line")
        macd_signal = st.number_input("MACD Signal", value=0.0, step=0.001, format="%.4f", help="MACD signal line")
        bb_position = st.slider("Bollinger Band Position", 0.0, 1.0, 0.5, 0.01, help="Price position relative to bands")
    
    with col2:
        st.markdown("**Additional Indicators**")
        sma_200 = st.number_input("SMA 200", value=0.0, step=0.01, format="%.2f", help="200-period Simple Moving Average")
        ema_9 = st.number_input("EMA 9", value=0.0, step=0.01, format="%.2f", help="9-period Exponential Moving Average")
        volume_ratio = st.number_input("Volume Ratio", min_value=0.0, value=1.0, step=0.1, format="%.2f", help="Current vs average volume")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.02, step=0.001, format="%.4f", help="Price volatility")
        trend_score = st.slider("Trend Score", -10.0, 10.0, 0.0, 0.1, help="Price momentum indicator")
        price_change_1h = st.number_input("1H Price Change (%)", value=0.0, step=0.1, format="%.2f", help="1-hour price change")
        price_change_4h = st.number_input("4H Price Change (%)", value=0.0, step=0.1, format="%.2f", help="4-hour price change")
        price_change_24h = st.number_input("24H Price Change (%)", value=0.0, step=0.1, format="%.2f", help="24-hour price change")
        
        st.markdown("**Feedback Summary**")
        st.markdown(f"- **Symbol:** {feedback_symbol}")
        st.markdown(f"- **Outcome:** {feedback_outcome}")
        st.markdown(f"- **P&L:** ${profit_loss:.2f}")
        st.markdown(f"- **Exchange:** {current_exchange.title()}")
        
        if st.button("💾 Add Feedback", type="primary", use_container_width=True):
            if not feedback_symbol.strip():
                st.error("Symbol cannot be empty")
            elif not feedback_symbol.isalnum():
                st.error("Symbol must contain only letters and numbers")
            else:
                try:
                    signal_data = {
                        'symbol': feedback_symbol,
                        'indicators': {
                            'rsi': float(rsi),
                            'stoch_rsi_k': float(stoch_rsi_k),
                            'stoch_rsi_d': float(stoch_rsi_d),
                            'macd': float(macd),
                            'macd_signal': float(macd_signal),
                            'macd_histogram': float(macd - macd_signal),
                            'bb_position': float(bb_position),
                            'sma_200': float(sma_200),
                            'ema_9': float(ema_9),
                            'volume_ratio': float(volume_ratio),
                            'volatility': float(volatility),
                            'trend_score': float(trend_score),
                            'price_change_1h': float(price_change_1h),
                            'price_change_4h': float(price_change_4h),
                            'price_change_24h': float(price_change_24h)
                        }
                    }
                    outcome = feedback_outcome.startswith("Success")
                    feedback_entry = FeedbackModel(
                        symbol=feedback_symbol,
                        outcome=outcome,
                        profit_loss=float(profit_loss),
                        signal_data=signal_data,
                        indicators=signal_data['indicators'],
                        exchange=current_exchange,
                        user_id=user_id,
                        timestamp=datetime.now(timezone.utc)
                    )
                    success = db_manager.add_feedback(feedback_entry)
                    if success:
                        st.success("✅ Feedback added successfully!")
                        ml_filter.update_model_with_feedback(signal_data, outcome)
                        st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add feedback")
                except Exception as e:
                    logger.error(f"Error adding feedback: {e}")
                    st.error(f"Error adding feedback: {e}")

# Bulk Feedback Management
st.divider()
st.markdown("""
### 🔄 Bulk Feedback Management
**Instructions:**
- Analyze all trades to generate feedback automatically.
- Clear feedback older than 30 days to keep data relevant.
- Export feedback data as a CSV file for external analysis.
""")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Analyze All Trades", type="primary", use_container_width=True):
        with st.spinner("Analyzing trades for feedback..."):
            try:
                trades = db_manager.get_trades(limit=1000, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
                trades_list = [t.to_dict() for t in trades]
                valid_trades = [t for t in trades_list if t['pnl'] is not None and t['indicators']]
                
                feedback_count = 0
                for trade in valid_trades:
                    try:
                        signal_data = {
                            'symbol': trade['symbol'],
                            'indicators': trade['indicators'],
                            'pnl': trade['pnl']
                        }
                        outcome = trade['pnl'] > 0
                        feedback_entry = FeedbackModel(
                            signal_id=trade['id'],
                            symbol=trade['symbol'],
                            outcome=outcome,
                            profit_loss=float(trade['pnl']),
                            signal_data=signal_data,
                            indicators=trade['indicators'],
                            exchange=trade['exchange'] or current_exchange,
                            user_id=user_id,
                            timestamp=datetime.now(timezone.utc)
                        )
                        if db_manager.add_feedback(feedback_entry):
                            feedback_count += 1
                    except Exception as e:
                        logger.error(f"Error adding bulk feedback for trade {trade['id']}: {e}")
                        continue
                
                st.success(f"✅ Added {feedback_count} feedback entries")
                st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()
            
            except Exception as e:
                logger.error(f"Error in bulk analysis: {e}")
                st.error(f"Error in bulk analysis: {e}")

with col2:
    if st.button("🗑️ Clear Old Feedback", type="secondary", use_container_width=True):
        if st.button("⚠️ Confirm Clear", key="confirm_clear_feedback", type="secondary", use_container_width=True):
            try:
                feedback_data = db_manager.get_feedback(limit=1000, exchange=current_exchange, user_id=user_id)
                feedback_list = [f.to_dict() for f in feedback_data]
                older_than_30_days = [
                    f for f in feedback_list 
                    if (datetime.fromisoformat(f['timestamp']) if isinstance(f['timestamp'], str) else f['timestamp']) < 
                    datetime.now(timezone.utc) - timedelta(days=30)
                ]
                if older_than_30_days:
                    with db_manager.get_session() as session:
                        for feedback in older_than_30_days:
                            session.query(FeedbackModel).filter_by(id=feedback['id'], user_id=user_id).delete()
                        session.commit()
                    st.success(f"✅ Cleared {len(older_than_30_days)} old feedback entries")
                    st.session_state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
                else:
                    st.info("No old feedback entries to clear")
            except Exception as e:
                logger.error(f"Error clearing feedback: {e}")
                st.error(f"Error clearing feedback: {e}")

with col3:
    if st.button("📤 Export Feedback", type="secondary", use_container_width=True):
        try:
            feedback_data = db_manager.get_feedback(limit=1000, exchange=current_exchange, user_id=user_id)
            if feedback_data:
                export_data = [{
                    'ID': f.id,
                    'Symbol': f.symbol,
                    'Outcome': f.outcome,
                    'Profit_Loss': f.profit_loss,
                    'Exchange': f.exchange,
                    'Timestamp': (datetime.fromisoformat(f.timestamp) if isinstance(f.timestamp, str) else f.timestamp).isoformat(),
                    'Signal_Data': json.dumps(f.signal_data),
                    'Indicators': json.dumps(f.indicators)
                } for f in feedback_data]
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "💾 Download Feedback Data",
                    data=csv,
                    file_name=f"ml_feedback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No feedback data to export")
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            st.error(f"Error exporting feedback: {e}")

# Status Footer
st.divider()
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <strong>ML System Status:</strong> Exchange: {current_exchange.title()} | 
    Mode: {account_type.title()} | 
    User: {username} | 
    Model: {'✅ Active' if ml_filter.model is not None else '❌ Inactive'} | 
    Last Updated: {st.session_state.get('last_updated', 'N/A')}
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass