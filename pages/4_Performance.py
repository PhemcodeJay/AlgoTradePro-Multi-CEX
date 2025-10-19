import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timezone, timedelta
from db import db_manager, User, WalletBalance
from logging_config import get_trading_logger
import bcrypt

# Page configuration
st.set_page_config(
    page_title="Performance - AlgoTraderPro V2.0",
    page_icon="📊",
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

st.title("📊 Performance Analytics")
st.markdown(f"**Exchange:** {current_exchange.title()} | **Mode:** {account_type.title()}")

# Performance overview
st.subheader("🎯 Portfolio Performance")

try:
    stats = trading_engine.get_trade_statistics(account_type) or {}
    
    wallet = db_manager.get_wallet_balance(account_type, user_id=user_id, exchange=current_exchange)
    current_balance = wallet.get('available', 100.0 if account_type == 'virtual' else 0.0)
    initial_balance = wallet.get('total', 100.0) if account_type == 'virtual' else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = current_balance - initial_balance
        return_pct = (total_return / initial_balance) * 100 if initial_balance > 0 else 0
        st.metric("Portfolio Value", f"${current_balance:.2f}", delta=f"{return_pct:+.2f}%")
    
    with col2:
        st.metric("Total P&L", f"${stats.get('total_pnl', 0):.2f}")
    
    with col3:
        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
    
    with col4:
        st.metric("Total Trades", stats.get('total_trades', 0))

except Exception as e:
    logger.error(f"Error loading performance overview: {e}")
    st.error(f"Error loading performance overview: {e}")

st.divider()

# Detailed performance analysis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["P&L Analysis", "Risk Metrics", "Trade Analysis", "Symbol Performance", "ML Performance"])

with tab1:
    st.subheader("💹 Profit & Loss Analysis")
    
    try:
        closed_trades = db_manager.get_trades(
            status='closed',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )
        
        if closed_trades:
            trade_data = [trade.to_dict() for trade in closed_trades]
            trade_dates = []
            cumulative_pnl = []
            daily_pnl = []
            
            running_pnl = 0
            for trade in sorted(trade_data, key=lambda x: x.get('updated_at') or x.get('created_at')):
                if trade.get('pnl') is not None:
                    trade_date = trade.get('updated_at') or trade.get('created_at')
                    if isinstance(trade_date, str):
                        trade_date = datetime.fromisoformat(trade_date)
                    trade_dates.append(trade_date)
                    running_pnl += trade['pnl']
                    cumulative_pnl.append(running_pnl)
                    daily_pnl.append(trade['pnl'])
            
            if trade_dates:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=trade_dates,
                        y=cumulative_pnl,
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='blue', width=2),
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.1)'
                    ))
                    fig_cum.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                    fig_cum.update_layout(
                        title="Cumulative P&L Over Time",
                        xaxis_title="Date",
                        yaxis_title="P&L ($)",
                        height=400
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                with col2:
                    fig_dist = px.histogram(
                        x=daily_pnl,
                        nbins=min(30, len(daily_pnl)),
                        title="Trade P&L Distribution",
                        color_discrete_sequence=['green']
                    )
                    fig_dist.update_layout(
                        xaxis_title="P&L ($)",
                        yaxis_title="Number of Trades",
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                st.subheader("📈 Key Performance Indicators")
                profitable_trades = [p for p in daily_pnl if p > 0]
                losing_trades = [p for p in daily_pnl if p < 0]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Trade", f"${max(daily_pnl):.2f}" if daily_pnl else "$0.00")
                    st.metric("Worst Trade", f"${min(daily_pnl):.2f}" if daily_pnl else "$0.00")
                
                with col2:
                    avg_win = sum(profitable_trades) / len(profitable_trades) if profitable_trades else 0
                    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
                    st.metric("Avg Win", f"${avg_win:.2f}")
                    st.metric("Avg Loss", f"${avg_loss:.2f}")
                
                with col3:
                    profit_factor = abs(sum(profitable_trades) / sum(losing_trades)) if losing_trades else float('inf')
                    st.metric("Profit Factor", f"{profit_factor:.2f}")
                    st.metric("Total Trades", len(daily_pnl))
                
                with col4:
                    win_rate = (len(profitable_trades) / len(daily_pnl)) * 100 if daily_pnl else 0
                    expectancy = sum(daily_pnl) / len(daily_pnl) if daily_pnl else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                    st.metric("Expectancy", f"${expectancy:.2f}")
                
                st.subheader("📅 Monthly Performance")
                monthly_data = {}
                for i, trade_date in enumerate(trade_dates):
                    month_key = trade_date.strftime('%Y-%m')
                    monthly_data.setdefault(month_key, []).append(daily_pnl[i])
                
                if monthly_data:
                    monthly_summary = []
                    for month, pnls in monthly_data.items():
                        monthly_summary.append({
                            'Month': month,
                            'Total P&L': f"${sum(pnls):.2f}",
                            'Trades': len(pnls),
                            'Win Rate': f"{(len([p for p in pnls if p > 0]) / len(pnls) * 100):.1f}%",
                            'Best Trade': f"${max(pnls):.2f}",
                            'Worst Trade': f"${min(pnls):.2f}"
                        })
                    
                    df_monthly = pd.DataFrame(monthly_summary)
                    st.dataframe(df_monthly, use_container_width=True)
        
        else:
            st.info("No closed trades available for P&L analysis")
    
    except Exception as e:
        logger.error(f"Error in P&L analysis: {e}")
        st.error(f"Error in P&L analysis: {e}")

with tab2:
    st.subheader("⚠️ Risk Metrics")
    
    try:
        closed_trades = db_manager.get_trades(
            status='closed',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )
        
        if closed_trades:
            trade_data = [trade.to_dict() for trade in closed_trades]
            pnl_values = [trade['pnl'] for trade in trade_data if trade['pnl'] is not None]
            
            if pnl_values:
                cumulative_pnl = np.cumsum(pnl_values)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = cumulative_pnl - running_max
                max_drawdown = np.min(drawdown)
                
                std_dev = np.std(pnl_values)
                downside_returns = [p for p in pnl_values if p < 0]
                downside_deviation = np.std(downside_returns) if downside_returns else 0
                avg_return = np.mean(pnl_values)
                sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Drawdown", f"${max_drawdown:.2f}")
                    st.metric("Std Deviation", f"${std_dev:.2f}")
                
                with col2:
                    st.metric("Downside Deviation", f"${downside_deviation:.2f}")
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                
                with col3:
                    var_95 = np.percentile(pnl_values, 5)
                    st.metric("VaR (95%)", f"${var_95:.2f}")
                    consecutive_losses = 0
                    max_consecutive_losses = 0
                    for pnl in pnl_values:
                        if pnl < 0:
                            consecutive_losses += 1
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        else:
                            consecutive_losses = 0
                    st.metric("Max Consecutive Losses", max_consecutive_losses)
                
                with col4:
                    risk_adj_return = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0
                    st.metric("Risk-Adj Return", f"{risk_adj_return:.3f}")
                    recovery_factor = sum(pnl_values) / abs(max_drawdown) if max_drawdown != 0 else 0
                    st.metric("Recovery Factor", f"{recovery_factor:.2f}")
                
                st.subheader("📉 Drawdown Analysis")
                drawdown_dates = [trade.get('updated_at') or trade.get('created_at') 
                                 for trade in sorted(trade_data, key=lambda x: x.get('updated_at') or x.get('created_at'))
                                 if trade.get('pnl') is not None]
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown_dates,
                    y=drawdown,
                    mode='lines',
                    fill='tonexty',
                    name='Drawdown',
                    line=dict(color='red'),
                    fillcolor='rgba(255,0,0,0.3)'
                ))
                fig_dd.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                fig_dd.update_layout(
                    title="Portfolio Drawdown Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown ($)",
                    height=400
                )
                st.plotly_chart(fig_dd, use_container_width=True)
                
                st.subheader("⏱️ Risk by Trade Duration")
                duration_risk = []
                for trade in trade_data:
                    if trade.get('pnl') is not None and trade.get('created_at') and trade.get('updated_at'):
                        if isinstance(trade['created_at'], str):
                            trade['created_at'] = datetime.fromisoformat(trade['created_at'])
                        if isinstance(trade['updated_at'], str):
                            trade['updated_at'] = datetime.fromisoformat(trade['updated_at'])
                        duration_hours = (trade['updated_at'] - trade['created_at']).total_seconds() / 3600
                        duration_risk.append({
                            'Duration (Hours)': duration_hours,
                            'P&L': trade['pnl']
                        })
                
                if duration_risk:
                    df_duration_risk = pd.DataFrame(duration_risk)
                    fig_duration = px.scatter(
                        df_duration_risk,
                        x='Duration (Hours)',
                        y='P&L',
                        color='P&L',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title="P&L vs Trade Duration"
                    )
                    fig_duration.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                    st.plotly_chart(fig_duration, use_container_width=True)
        
        else:
            st.info("No closed trades available for risk analysis")
    
    except Exception as e:
        logger.error(f"Error in risk analysis: {e}")
        st.error(f"Error in risk analysis: {e}")

with tab3:
    st.subheader("🔍 Trade Analysis")
    
    try:
        closed_trades = db_manager.get_trades(
            status='closed',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )
        
        if closed_trades:
            trade_data = [trade.to_dict() for trade in closed_trades]
            trade_sizes = []
            trade_returns = []
            sides = []
            
            for trade in trade_data:
                if trade.get('pnl') is not None and trade.get('entry_price') and trade.get('qty'):
                    trade_value = trade['entry_price'] * trade['qty']
                    return_pct = (trade['pnl'] / trade_value) * 100 if trade_value > 0 else 0
                    trade_sizes.append(trade_value)
                    trade_returns.append(return_pct)
                    sides.append(trade['side'])
            
            if trade_sizes:
                col1, col2 = st.columns(2)
                with col1:
                    fig_size = px.histogram(
                        x=trade_sizes,
                        nbins=20,
                        title="Trade Size Distribution"
                    )
                    fig_size.update_layout(
                        xaxis_title="Trade Value ($)",
                        yaxis_title="Number of Trades"
                    )
                    st.plotly_chart(fig_size, use_container_width=True)
                
                with col2:
                    fig_returns = px.histogram(
                        x=trade_returns,
                        nbins=20,
                        title="Trade Return Distribution (%)"
                    )
                    fig_returns.update_layout(
                        xaxis_title="Return (%)",
                        yaxis_title="Number of Trades"
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                st.write("**Performance by Trade Side:**")
                side_performance = {}
                for i, side in enumerate(sides):
                    side_performance.setdefault(side, []).append(trade_returns[i])
                
                side_summary = []
                for side, returns in side_performance.items():
                    side_summary.append({
                        'Side': side,
                        'Trades': len(returns),
                        'Avg Return': f"{np.mean(returns):.2f}%",
                        'Win Rate': f"{(len([r for r in returns if r > 0]) / len(returns) * 100):.1f}%",
                        'Best Trade': f"{max(returns):.2f}%",
                        'Worst Trade': f"{min(returns):.2f}%"
                    })
                
                df_sides = pd.DataFrame(side_summary)
                st.dataframe(df_sides, use_container_width=True)
                
                st.write("**Trade Timing Analysis:**")
                hour_performance = {}
                day_performance = {}
                
                for trade in trade_data:
                    if trade.get('pnl') is not None and trade.get('created_at'):
                        if isinstance(trade['created_at'], str):
                            trade['created_at'] = datetime.fromisoformat(trade['created_at'])
                        trade_time = trade['created_at']
                        hour = trade_time.hour
                        day = trade_time.strftime('%A')
                        return_pct = (trade['pnl'] / (trade['entry_price'] * trade['qty']) * 100) if trade.get('entry_price') and trade.get('qty') else 0
                        hour_performance.setdefault(hour, []).append(return_pct)
                        day_performance.setdefault(day, []).append(return_pct)
                
                col1, col2 = st.columns(2)
                with col1:
                    if hour_performance:
                        hours = sorted(hour_performance.keys())
                        avg_hourly_returns = [np.mean(hour_performance[h]) for h in hours]
                        fig_hourly = px.bar(
                            x=hours,
                            y=avg_hourly_returns,
                            title="Average Return by Hour of Day"
                        )
                        fig_hourly.update_layout(
                            xaxis_title="Hour",
                            yaxis_title="Average Return (%)"
                        )
                        st.plotly_chart(fig_hourly, use_container_width=True)
                
                with col2:
                    if day_performance:
                        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        avg_daily_returns = [np.mean(day_performance.get(d, [0])) for d in days]
                        fig_daily = px.bar(
                            x=days,
                            y=avg_daily_returns,
                            title="Average Return by Day of Week"
                        )
                        fig_daily.update_layout(
                            xaxis_title="Day",
                            yaxis_title="Average Return (%)"
                        )
                        st.plotly_chart(fig_daily, use_container_width=True)
        
        else:
            st.info("No closed trades available for trade analysis")
    
    except Exception as e:
        logger.error(f"Error in trade analysis: {e}")
        st.error(f"Error in trade analysis: {e}")

with tab4:
    st.subheader("🏷️ Symbol Performance")
    
    try:
        closed_trades = db_manager.get_trades(
            status='closed',
            virtual=(account_type == 'virtual'),
            exchange=current_exchange,
            user_id=user_id,
            limit=2000
        )
        
        if closed_trades:
            trade_data = [trade.to_dict() for trade in closed_trades]
            symbol_stats = {}
            for trade in trade_data:
                if trade.get('pnl') is not None:
                    symbol = trade['symbol']
                    symbol_stats.setdefault(symbol, {
                        'trades': 0,
                        'total_pnl': 0,
                        'wins': 0,
                        'losses': 0,
                        'pnl_list': []
                    })
                    symbol_stats[symbol]['trades'] += 1
                    symbol_stats[symbol]['total_pnl'] += trade['pnl']
                    symbol_stats[symbol]['pnl_list'].append(trade['pnl'])
                    if trade['pnl'] > 0:
                        symbol_stats[symbol]['wins'] += 1
                    elif trade['pnl'] < 0:
                        symbol_stats[symbol]['losses'] += 1
            
            if symbol_stats:
                symbol_summary = []
                for symbol, stats in symbol_stats.items():
                    win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                    avg_pnl = stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
                    symbol_summary.append({
                        'Symbol': symbol,
                        'Trades': stats['trades'],
                        'Total P&L': f"${stats['total_pnl']:.2f}",
                        'Avg P&L': f"${avg_pnl:.2f}",
                        'Win Rate': f"{win_rate:.1f}%",
                        'Best Trade': f"${max(stats['pnl_list']):.2f}",
                        'Worst Trade': f"${min(stats['pnl_list']):.2f}"
                    })
                
                symbol_summary.sort(key=lambda x: float(x['Total P&L'].replace('$', '').replace(',', '')), reverse=True)
                df_symbols = pd.DataFrame(symbol_summary)
                
                def color_pnl(val):
                    try:
                        val_float = float(val.replace('$', ''))
                        return 'background-color: rgba(255, 0, 0, 0.3)' if val_float < 0 else \
                               'background-color: rgba(0, 255, 0, 0.3)' if val_float > 0 else ''
                    except:
                        return ''
                
                styled_df = df_symbols.style.applymap(color_pnl, subset=['Total P&L'])
                st.dataframe(styled_df, use_container_width=True)
                
                st.write("**Top/Bottom Performers:**")
                symbols = [s['Symbol'] for s in symbol_summary[:10]]
                total_pnls = [float(s['Total P&L'].replace('$', '').replace(',', '')) for s in symbol_summary[:10]]
                
                fig_performers = px.bar(
                    x=symbols,
                    y=total_pnls,
                    title="Symbol Performance (Total P&L)",
                    color=total_pnls,
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig_performers.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                fig_performers.update_layout(
                    xaxis_title="Symbol",
                    yaxis_title="Total P&L ($)",
                    height=400
                )
                st.plotly_chart(fig_performers, use_container_width=True)
                
                if len(symbol_stats) > 1:
                    st.write("**Symbol Return Correlations:**")
                    symbol_returns = {}
                    max_trades = max(len(stats['pnl_list']) for stats in symbol_stats.values())
                    
                    for symbol, stats in symbol_stats.items():
                        if len(stats['pnl_list']) >= 5:
                            returns = [pnl / 100 for pnl in stats['pnl_list'][:20]]  # Simplified returns
                            symbol_returns[symbol] = returns
                    
                    if len(symbol_returns) > 1:
                        df_corr = pd.DataFrame.from_dict(symbol_returns, orient='index').T
                        correlation_matrix = df_corr.corr()
                        fig_corr = px.imshow(
                            correlation_matrix,
                            title="Symbol Return Correlations",
                            color_continuous_scale='RdBu',
                            aspect='auto'
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
        
        else:
            st.info("No closed trades available for symbol analysis")
    
    except Exception as e:
        logger.error(f"Error in symbol analysis: {e}")
        st.error(f"Error in symbol analysis: {e}")

with tab5:
    st.subheader("🤖 ML Performance")
    
    try:
        from ml import MLFilter
        ml_filter = MLFilter()
        
        if ml_filter.model is not None:
            st.success("✅ ML Model is loaded and active")
            
            importance = ml_filter.get_feature_importance()
            if importance:
                st.write("**Feature Importance:**")
                features = list(importance.keys())
                importances = list(importance.values())
                
                fig_importance = px.bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    title="ML Feature Importance",
                    color=importances,
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                importance_df = pd.DataFrame([
                    {'Feature': feature, 'Importance': f"{importance:.4f}"}
                    for feature, importance in importance.items()
                ])
                st.dataframe(importance_df, use_container_width=True)
            
            st.write("**ML Feedback Analysis:**")
            feedback_data = db_manager.get_feedback(
                limit=100,
                exchange=current_exchange,
                user_id=user_id
            )
            
            if feedback_data:
                feedback_list = [f.to_dict() for f in feedback_data]
                positive_feedback = len([f for f in feedback_list if f.get('outcome')])
                total_feedback = len(feedback_list)
                ml_accuracy = (positive_feedback / total_feedback) * 100 if total_feedback > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Feedback", total_feedback)
                with col2:
                    st.metric("Positive Outcomes", positive_feedback)
                with col3:
                    st.metric("ML Accuracy", f"{ml_accuracy:.1f}%")
                with col4:
                    recent_feedback = len([f for f in feedback_list 
                                         if isinstance(f.get('timestamp'), datetime) and 
                                         f['timestamp'] > datetime.now(timezone.utc) - timedelta(days=7)])
                    st.metric("Recent Feedback", recent_feedback)
                
                if len(feedback_list) > 5:
                    feedback_dates = [f['timestamp'] for f in feedback_list if isinstance(f.get('timestamp'), datetime)]
                    feedback_outcomes = [1 if f.get('outcome') else 0 for f in feedback_list]
                    
                    window_size = min(10, len(feedback_list) // 2)
                    rolling_accuracy = []
                    for i in range(window_size - 1, len(feedback_outcomes)):
                        window_outcomes = feedback_outcomes[i - window_size + 1:i + 1]
                        accuracy = sum(window_outcomes) / len(window_outcomes) * 100
                        rolling_accuracy.append(accuracy)
                    
                    if rolling_accuracy and feedback_dates:
                        fig_ml_trend = go.Figure()
                        fig_ml_trend.add_trace(go.Scatter(
                            x=feedback_dates[window_size - 1:],
                            y=rolling_accuracy,
                            mode='lines+markers',
                            name=f'Rolling Accuracy ({window_size}-trade window)',
                            line=dict(color='purple', width=2)
                        ))
                        fig_ml_trend.add_hline(y=50, line_dash="dash", line_color="red", 
                                             annotation_text="Random Baseline (50%)")
                        fig_ml_trend.update_layout(
                            title="ML Model Accuracy Trend",
                            xaxis_title="Date",
                            yaxis_title="Accuracy (%)",
                            height=400
                        )
                        st.plotly_chart(fig_ml_trend, use_container_width=True)
                
                st.write("**ML vs. Traditional Signal Performance:**")
                recent_signals = db_manager.get_signals(
                    limit=100,
                    exchange=current_exchange,
                    user_id=user_id
                )
                ml_enhanced_signals = []
                traditional_signals = []
                
                for signal in recent_signals:
                    signal_dict = signal.to_dict()
                    indicators = signal_dict.get('indicators', {})
                    if 'ml_score' in indicators:
                        ml_enhanced_signals.append(signal_dict)
                    else:
                        traditional_signals.append(signal_dict)
                
                if ml_enhanced_signals or traditional_signals:
                    comparison_data = []
                    if ml_enhanced_signals:
                        avg_ml_score = np.mean([s['score'] for s in ml_enhanced_signals])
                        comparison_data.append({'Type': 'ML Enhanced', 'Avg Score': avg_ml_score, 'Count': len(ml_enhanced_signals)})
                    if traditional_signals:
                        avg_traditional_score = np.mean([s['score'] for s in traditional_signals])
                        comparison_data.append({'Type': 'Traditional', 'Avg Score': avg_traditional_score, 'Count': len(traditional_signals)})
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_comparison = px.bar(
                                df_comparison,
                                x='Type',
                                y='Avg Score',
                                title="Average Signal Score Comparison"
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        with col2:
                            fig_count = px.bar(
                                df_comparison,
                                x='Type',
                                y='Count',
                                title="Signal Count Comparison"
                            )
                            st.plotly_chart(fig_count, use_container_width=True)
            
            else:
                st.info("No ML feedback data available")
        
        else:
            st.warning("⚠️ ML Model is not loaded")
            st.info("Train the ML model from the ML Feedback page to see ML performance metrics.")
    
    except Exception as e:
        logger.error(f"Error in ML performance analysis: {e}")
        st.error(f"Error in ML performance analysis: {e}")

# Export functionality
st.divider()
st.subheader("📤 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Performance Report"):
        try:
            stats = trading_engine.get_trade_statistics(account_type)
            report_data = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'exchange': current_exchange,
                'account_type': account_type,
                'user_id': user_id,
                'statistics': stats
            }
            st.download_button(
                "💾 Download Report (JSON)",
                data=pd.Series(report_data).to_json(indent=2),
                file_name=f"performance_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            st.error(f"Error generating report: {e}")

with col2:
    if st.button("📈 Export Trade Data"):
        try:
            closed_trades = db_manager.get_trades(
                status='closed',
                virtual=(account_type == 'virtual'),
                exchange=current_exchange,
                user_id=user_id,
                limit=2000
            )
            if closed_trades:
                trade_data = [{
                    'ID': trade.id,
                    'Symbol': trade.symbol,
                    'Side': trade.side,
                    'Quantity': trade.qty,
                    'Entry Price': trade.entry_price,
                    'Exit Price': trade.exit_price,
                    'P&L': trade.pnl,
                    'Created': trade.created_at.isoformat() if isinstance(trade.created_at, datetime) else trade.created_at,
                    'Updated': trade.updated_at.isoformat() if isinstance(trade.updated_at, datetime) else trade.updated_at
                } for trade in closed_trades]
                
                df_trades = pd.DataFrame(trade_data)
                csv = df_trades.to_csv(index=False)
                st.download_button(
                    "💾 Download Trades (CSV)",
                    data=csv,
                    file_name=f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No trade data to export")
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            st.error(f"Error exporting trades: {e}")

with col3:
    if st.button("🤖 Export ML Data"):
        try:
            feedback_data = db_manager.get_feedback(
                limit=1000,
                exchange=current_exchange,
                user_id=user_id
            )
            if feedback_data:
                ml_data = [{
                    'ID': f.id,
                    'Symbol': f.symbol,
                    'Outcome': f.outcome,
                    'Profit/Loss': f.profit_loss,
                    'Exchange': f.exchange,
                    'Timestamp': f.timestamp.isoformat() if isinstance(f.timestamp, datetime) else f.timestamp
                } for f in feedback_data]
                
                df_ml = pd.DataFrame(ml_data)
                csv = df_ml.to_csv(index=False)
                st.download_button(
                    "💾 Download ML Data (CSV)",
                    data=csv,
                    file_name=f"ml_feedback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No ML feedback data to export")
        except Exception as e:
            logger.error(f"Error exporting ML data: {e}")
            st.error(f"Error exporting ML data: {e}")

# Status footer
st.divider()
st.markdown(f"""
**Status:** Exchange: {current_exchange.title()} | 
Mode: {account_type.title()} | 
Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
""")