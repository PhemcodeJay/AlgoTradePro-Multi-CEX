import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timezone, timedelta
from db import db_manager, FeedbackModel, User, WalletBalance
from ml import MLFilter
from logging_config import get_trading_logger
import bcrypt

# Page configuration
st.set_page_config(
    page_title="AI Analysis - AlgoTraderPro V2.0",
    page_icon="🤖",
    layout="wide"
)

# Initialize logger
logger = get_trading_logger(__name__)

# Initialize session state
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

# Initialize components
if 'trading_engine' not in st.session_state or st.session_state.trading_engine is None:
    st.error("Trading engine not initialized. Please return to the main page to initialize the application.")
    st.stop()

trading_engine = st.session_state.trading_engine
current_exchange = st.session_state.get('current_exchange', 'binance')
account_type = st.session_state.get('account_type', 'virtual')
user_id = st.session_state.user.get('id') if st.session_state.user else None

if not user_id:
    st.error("User not authenticated. Please log in from the main page.")
    st.stop()

st.title("🤖 Machine Learning Feedback & Training")
st.markdown(f"**Exchange:** {current_exchange.title()} | **Mode:** {account_type.title()} | **User:** {st.session_state.user.get('username', 'N/A')}")

# ML Status Overview
st.subheader("📊 ML Model Status")

try:
    ml_filter = MLFilter()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        model_status = "✅ Loaded" if ml_filter.model is not None else "❌ Not Loaded"
        st.metric("Model Status", model_status)
    
    with col2:
        feedback_count = len(db_manager.get_feedback(limit=1000, exchange=current_exchange, user_id=user_id))
        st.metric("Total Feedback", feedback_count)
    
    with col3:
        recent_feedback = len(db_manager.get_feedback(limit=100, exchange=current_exchange, user_id=user_id))
        st.metric(f"{current_exchange.title()} Feedback", recent_feedback)
    
    with col4:
        recent_data = db_manager.get_feedback(limit=100, exchange=current_exchange, user_id=user_id)
        if recent_data:
            recent_data_list = [f.to_dict() for f in recent_data]
            positive_outcomes = len([f for f in recent_data_list if f['outcome']])
            accuracy = (positive_outcomes / len(recent_data_list)) * 100
            st.metric("Recent Accuracy", f"{accuracy:.1f}%")
        else:
            st.metric("Recent Accuracy", "N/A")

except Exception as e:
    logger.error(f"Error loading ML status: {e}")
    st.error(f"Error loading ML status: {e}")

st.divider()

# Helper functions
def get_feature_description(feature: str) -> str:
    """Get description for ML features"""
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
        'price_change_24h': '24-hour price change percentage'
    }
    return descriptions.get(feature, 'Technical indicator')

def get_feature_performance(feature: str) -> str:
    """Get performance metrics for a specific feature"""
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
        
        return f"Correlation with outcome: {correlation:.3f}, Profitable mean: {profitable_mean:.3f}, Unprofitable mean: {unprofitable_mean:.3f}"
    
    except Exception as e:
        logger.error(f"Error calculating feature performance for {feature}: {e}")
        return "Error calculating performance"

def get_feature_suggestion(feature: str) -> str:
    """Get improvement suggestions for features"""
    suggestions = {
        'rsi': 'Consider adding RSI divergence signals and different timeframe RSI values',
        'macd': 'Combine with MACD crossover signals and histogram analysis',
        'bb_position': 'Add Bollinger Band squeeze detection and expansion indicators',
        'volume_ratio': 'Include volume profile analysis and volume-price trend correlation',
        'volatility': 'Consider volatility breakouts and volatility regime changes',
        'trend_score': 'Enhance with multiple timeframe trend alignment',
        'price_change_1h': 'Add momentum acceleration and deceleration signals',
        'price_change_4h': 'Combine with support/resistance level interactions',
        'price_change_24h': 'Include market structure analysis and swing points'
    }
    return suggestions.get(feature, 'Consider additional technical analysis indicators')

# ML Management Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Training", "Feedback Analysis", "Feature Importance", "Model Performance", "Manual Feedback"])

with tab1:
    st.subheader("🏋️ Model Training & Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Training Data Overview:**")
        
        try:
            all_trades = db_manager.get_trades(limit=1000, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
            all_trades_list = [t.to_dict() for t in all_trades]
            virtual_trades = [t for t in all_trades_list if t['pnl'] is not None]
            
            if virtual_trades:
                profitable_trades = len([t for t in virtual_trades if t['pnl'] > 0])
                
                training_stats = {
                    "Total Trades": len(virtual_trades),
                    "Profitable": profitable_trades,
                    "Unprofitable": len(virtual_trades) - profitable_trades,
                    "Success Rate": f"{(profitable_trades / len(virtual_trades) * 100):.1f}%"
                }
                
                for key, value in training_stats.items():
                    st.write(f"**{key}:** {value}")
                
                trades_with_indicators = [t for t in virtual_trades if t['indicators']]
                indicator_coverage = (len(trades_with_indicators) / len(virtual_trades)) * 100
                
                st.progress(indicator_coverage / 100, text=f"Indicator Coverage: {indicator_coverage:.1f}%")
                
                if indicator_coverage < 80:
                    st.warning("⚠️ Low indicator coverage may affect ML model quality")
                elif indicator_coverage >= 95:
                    st.success("✅ Excellent indicator coverage for training")
                
                recent_trades = [t for t in virtual_trades 
                               if isinstance(t['created_at'], (datetime, str)) and 
                               (datetime.fromisoformat(t['created_at']) if isinstance(t['created_at'], str) else t['created_at']) > 
                               datetime.now(timezone.utc) - timedelta(days=30)]
                
                if recent_trades:
                    st.write(f"**Recent Training Data (30 days):** {len(recent_trades)} trades")
                else:
                    st.warning("No recent training data available")
            
            else:
                st.info("No virtual trades available for training")
        
        except Exception as e:
            logger.error(f"Error loading training data overview: {e}")
            st.error(f"Error loading training data overview: {e}")
    
    with col2:
        st.write("**Model Actions:**")
        
        if st.button("🚀 Train ML Model", type="primary"):
            if len(virtual_trades) >= 50:
                with st.spinner("Training ML model..."):
                    try:
                        training_data = []
                        for trade in virtual_trades:
                            if trade['indicators'] and trade['pnl'] is not None:
                                training_data.append({
                                    'indicators': trade['indicators'],
                                    'pnl': trade['pnl']
                                })
                        
                        success = ml_filter.train_model(training_data)
                        
                        if success:
                            st.success("✅ Model trained successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Model training failed")
                    except Exception as e:
                        logger.error(f"Training error: {e}")
                        st.error(f"Training error: {e}")
            else:
                st.error(f"Need at least 50 trades for training. Current: {len(virtual_trades)}")
        
        if ml_filter.model is not None:
            st.write("**Current Model:**")
            st.write(f"Type: Random Forest")
            st.write(f"Features: {len(ml_filter.feature_columns)}")
            st.write(f"Exchange: {current_exchange}")
        
        if st.button("🗑️ Reset Model", type="secondary"):
            if st.button("⚠️ Confirm Reset", key="confirm_model_reset"):
                try:
                    model_path = f"ml_model_{current_exchange}_{user_id}.joblib"
                    scaler_path = f"ml_scaler_{current_exchange}_{user_id}.joblib"
                    
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                    
                    st.success("Model reset successfully!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error resetting model: {e}")
                    st.error(f"Error resetting model: {e}")

with tab2:
    st.subheader("📈 Feedback Analysis")
    
    try:
        feedback_data = db_manager.get_feedback(limit=500, exchange=current_exchange, user_id=user_id)
        
        if feedback_data:
            feedback_list = [f.to_dict() for f in feedback_data]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_count = len([f for f in feedback_list if f['outcome']])
                st.metric("Positive Outcomes", positive_count)
            
            with col2:
                negative_count = len([f for f in feedback_list if not f['outcome']])
                st.metric("Negative Outcomes", negative_count)
            
            with col3:
                accuracy = (positive_count / len(feedback_list)) * 100 if feedback_list else 0
                st.metric("Overall Accuracy", f"{accuracy:.1f}%")
            
            st.write("**Feedback Trends:**")
            
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
                    title="Daily Feedback Count"
                )
                st.plotly_chart(fig_feedback_count, use_container_width=True)
            
            with col2:
                fig_success_rate = px.line(
                    daily_feedback,
                    x='Date',
                    y='Success_Rate',
                    title="Daily Success Rate"
                )
                fig_success_rate.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                         annotation_text="Random Baseline (50%)")
                st.plotly_chart(fig_success_rate, use_container_width=True)
            
            st.write("**Performance by Symbol:**")
            
            symbol_stats = feedback_df.groupby('Symbol').agg({
                'Outcome': ['count', 'mean'],
                'Profit_Loss': 'sum'
            }).round(3)
            
            symbol_stats.columns = ['Feedback_Count', 'Success_Rate', 'Total_PnL']
            symbol_stats = symbol_stats.reset_index()
            symbol_stats = symbol_stats.sort_values('Total_PnL', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Performers:**")
                top_performers = symbol_stats.head(5)
                st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.write("**Bottom Performers:**")
                bottom_performers = symbol_stats.tail(5)
                st.dataframe(bottom_performers, use_container_width=True)
            
            st.write("**Recent Feedback (Last 20):**")
            
            recent_feedback_data = []
            for feedback in feedback_list[:20]:
                timestamp = datetime.fromisoformat(feedback['timestamp']) if isinstance(feedback['timestamp'], str) else feedback['timestamp']
                recent_feedback_data.append({
                    'Date': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': feedback['symbol'],
                    'Outcome': '✅ Success' if feedback['outcome'] else '❌ Failure',
                    'P&L': f"${feedback['profit_loss']:.2f}" if feedback['profit_loss'] else "N/A",
                    'Exchange': feedback['exchange']
                })
            
            df_recent = pd.DataFrame(recent_feedback_data)
            
            def color_outcome(val):
                if '✅' in val:
                    return 'background-color: rgba(0, 255, 0, 0.1)'
                elif '❌' in val:
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
    st.subheader("🎯 Feature Importance Analysis")
    
    try:
        ml_filter = MLFilter()
        
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
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(height=500)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.write("**Feature Importance Details:**")
                
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
                
                st.write("**Feature Analysis:**")
                
                recent_trades = db_manager.get_trades(limit=100, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
                recent_trades_list = [t.to_dict() for t in recent_trades]
                trades_with_indicators = [t for t in recent_trades_list if t['indicators']]
                
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
                        st.write("**Profitable Trades - Feature Stats:**")
                        st.dataframe(profitable_stats.round(4), height=300)
                    
                    with col2:
                        st.write("**Unprofitable Trades - Feature Stats:**")
                        st.dataframe(unprofitable_stats.round(4), height=300)
                
                st.write("**Feature Engineering Suggestions:**")
                
                top_features = list(importance.keys())[:3]
                suggestions = []
                
                for feature in top_features:
                    suggestion = get_feature_suggestion(feature)
                    suggestions.append(f"**{feature.upper()}:** {suggestion}")
                
                for suggestion in suggestions:
                    st.write(suggestion)
            
            else:
                st.warning("No feature importance data available")
        
        else:
            st.warning("No ML model loaded. Train a model first to see feature importance.")
    
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        st.error(f"Error in feature importance analysis: {e}")

with tab4:
    st.subheader("📊 Model Performance Metrics")
    
    try:
        ml_filter = MLFilter()
        
        if ml_filter.model is not None:
            st.write("**Model Validation:**")
            
            validation_trades = db_manager.get_trades(limit=200, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
            validation_trades_list = [t.to_dict() for t in validation_trades]
            virtual_validation = [t for t in validation_trades_list if t['indicators'] and t['pnl'] is not None]
            
            if len(virtual_validation) > 20:
                X_val = []
                y_true = []
                
                for trade in virtual_validation:
                    features = ml_filter.prepare_features(trade['indicators'])
                    if features.shape[1] > 0:
                        X_val.append(features.flatten())
                        y_true.append(1 if trade['pnl'] > 0 else 0)
                
                if len(X_val) > 10:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    X_val = np.array(X_val)
                    y_pred = ml_filter.model.predict(ml_filter.scaler.transform(X_val))
                    y_proba = ml_filter.model.predict_proba(ml_filter.scaler.transform(X_val))[:, 1]
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.3f}")
                    with col2:
                        st.metric("Precision", f"{precision:.3f}")
                    with col3:
                        st.metric("Recall", f"{recall:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.3f}")
                    
                    from sklearn.metrics import roc_curve, auc
                    
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
                    
                    st.write("**Prediction Distribution:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pred_dist = px.histogram(
                            x=y_proba,
                            nbins=20,
                            title="Prediction Probability Distribution"
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
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    st.write("**Performance by Confidence:**")
                    
                    confidence_data = []
                    for i, (prob, actual, pred) in enumerate(zip(y_proba, y_true, y_pred)):
                        confidence_data.append({
                            'Confidence': prob,
                            'Actual': actual,
                            'Predicted': pred,
                            'Correct': actual == pred
                        })
                    
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
                        title="Accuracy by Prediction Confidence"
                    )
                    fig_confidence.update_layout(xaxis_title="Confidence Range", yaxis_title="Accuracy")
                    st.plotly_chart(fig_confidence, use_container_width=True)
                
                else:
                    st.warning("Insufficient validation data for detailed metrics")
            
            else:
                st.info("Need more virtual trades for model validation")
        
        else:
            st.warning("No ML model loaded. Train a model first to see performance metrics.")
    
    except Exception as e:
        logger.error(f"Error in model performance analysis: {e}")
        st.error(f"Error in model performance analysis: {e}")

with tab5:
    st.subheader("✏️ Manual Feedback Entry")
    
    st.write("Add manual feedback to improve ML model performance:")
    
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
        
        st.write("**Signal Indicators:**")
        
        rsi = st.slider("RSI", 0.0, 100.0, 50.0, 0.1)
        macd = st.number_input("MACD", value=0.0, step=0.001, format="%.4f")
        macd_signal = st.number_input("MACD Signal", value=0.0, step=0.001, format="%.4f")
        bb_position = st.slider("Bollinger Band Position", 0.0, 1.0, 0.5, 0.01)
        volume_ratio = st.number_input("Volume Ratio", value=1.0, step=0.1, format="%.2f")
        volatility = st.number_input("Volatility", value=0.02, step=0.001, format="%.4f")
    
    with col2:
        st.write("**Additional Indicators:**")
        
        trend_score = st.slider("Trend Score", -10.0, 10.0, 0.0, 0.1)
        price_change_1h = st.number_input("1H Price Change (%)", value=0.0, step=0.1, format="%.2f")
        price_change_4h = st.number_input("4H Price Change (%)", value=0.0, step=0.1, format="%.2f")
        price_change_24h = st.number_input("24H Price Change (%)", value=0.0, step=0.1, format="%.2f")
        
        st.write("**Feedback Summary:**")
        st.write(f"**Symbol:** {feedback_symbol}")
        st.write(f"**Outcome:** {feedback_outcome}")
        st.write(f"**P&L:** ${profit_loss:.2f}")
        st.write(f"**Exchange:** {current_exchange}")
        
        if st.button("💾 Add Feedback", type="primary"):
            try:
                signal_data = {
                    'symbol': feedback_symbol,
                    'indicators': {
                        'rsi': rsi,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'macd_histogram': macd - macd_signal,
                        'bb_position': bb_position,
                        'volume_ratio': volume_ratio,
                        'volatility': volatility,
                        'trend_score': trend_score,
                        'price_change_1h': price_change_1h,
                        'price_change_4h': price_change_4h,
                        'price_change_24h': price_change_24h
                    }
                }
                
                outcome = feedback_outcome.startswith("Success")
                
                feedback_entry = FeedbackModel(
                    symbol=feedback_symbol,
                    outcome=outcome,
                    profit_loss=profit_loss,
                    signal_data=signal_data,
                    indicators=signal_data['indicators'],
                    exchange=current_exchange,
                    user_id=user_id
                )
                
                success = db_manager.add_feedback(feedback_entry)
                
                if success:
                    st.success("✅ Feedback added successfully!")
                    ml_filter.update_model_with_feedback(signal_data, outcome)
                    st.rerun()
                else:
                    st.error("❌ Failed to add feedback")
            
            except Exception as e:
                logger.error(f"Error adding feedback: {e}")
                st.error(f"Error adding feedback: {e}")

# Bulk feedback management
st.divider()
st.subheader("🔄 Bulk Feedback Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Analyze All Trades"):
        with st.spinner("Analyzing all trades for feedback..."):
            try:
                all_trades = db_manager.get_trades(limit=1000, exchange=current_exchange, user_id=user_id, virtual=(account_type == 'virtual'))
                all_trades_list = [t.to_dict() for t in all_trades]
                virtual_trades = [t for t in all_trades_list if t['pnl'] is not None and t['indicators']]
                
                feedback_count = 0
                for trade in virtual_trades:
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
                            profit_loss=trade['pnl'],
                            signal_data=signal_data,
                            indicators=trade['indicators'],
                            exchange=trade['exchange'] or current_exchange,
                            user_id=user_id
                        )
                        
                        if db_manager.add_feedback(feedback_entry):
                            feedback_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error adding bulk feedback for trade {trade['id']}: {e}")
                        continue
                
                st.success(f"✅ Added {feedback_count} feedback entries from trades")
                st.rerun()
            
            except Exception as e:
                logger.error(f"Error in bulk analysis: {e}")
                st.error(f"Error in bulk analysis: {e}")

with col2:
    if st.button("🗑️ Clear Old Feedback"):
        if st.button("⚠️ Confirm Clear", key="confirm_clear_feedback"):
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
                    st.rerun()
                else:
                    st.info("No old feedback entries to clear")
                
            except Exception as e:
                logger.error(f"Error clearing feedback: {e}")
                st.error(f"Error clearing feedback: {e}")

with col3:
    if st.button("📤 Export Feedback"):
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
                    mime="text/csv"
                )
            else:
                st.info("No feedback data to export")
        
        except Exception as e:
            logger.error(f"Error exporting feedback: {e}")
            st.error(f"Error exporting feedback: {e}")

# Status footer
st.divider()
st.markdown(f"""
**ML System Status:** Exchange: {current_exchange.title()} | 
Mode: {account_type.title()} | 
User: {st.session_state.user.get('username', 'N/A')} | 
Model: {'✅ Active' if ml_filter.model is not None else '❌ Inactive'} | 
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""")