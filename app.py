import streamlit as st
import os
from datetime import datetime
from db import db_manager, Feedback
from logging_config import get_logger
from dotenv import load_dotenv, set_key

# Logging using centralized system
logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AlgoTraderPro 2.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Env Setup ---
ENV_PATH = ".env"
load_dotenv(ENV_PATH)

# --- Session State Defaults ---
if "exchange" not in st.session_state:
    st.session_state.exchange = os.getenv("EXCHANGE", "binance").lower()

if "trading_modes" not in st.session_state:
    st.session_state.trading_modes = {}  # store per exchange {"binance": "virtual", "bybit": "real"}

if "engine_initialized" not in st.session_state:
    st.session_state.engine_initialized = False
if "wallet_cache" not in st.session_state:
    st.session_state.wallet_cache = {}
if "client" not in st.session_state:
    st.session_state.client = None
if "engine" not in st.session_state:
    st.session_state.engine = None

EXCHANGE = st.session_state.exchange


# --- Exchange Switch ---
def set_exchange(exchange: str):
    """Switch exchange in session state and persist in .env"""
    st.session_state.exchange = exchange.lower()
    set_key(ENV_PATH, "EXCHANGE", st.session_state.exchange)  # Update .env
    st.session_state.client = None
    st.session_state.engine = None
    st.session_state.engine_initialized = False
    st.session_state.wallet_cache.clear()
    st.rerun()


# --- Dynamic Imports ---
if EXCHANGE == "binance":
    from binance_client import BinanceClient
    from multi_trading_engine import TradingEngine
elif EXCHANGE == "bybit":
    from bybit_client import BybitClient
    from multi_trading_engine import TradingEngine
else:
    raise ValueError(f"Unsupported exchange: {EXCHANGE}")


# --- Initialize trading engine ---
def initialize_engine():
    try:
        if not st.session_state.engine_initialized:
            st.session_state.engine = TradingEngine()
            st.session_state.engine_initialized = True
            logger.info(f"{EXCHANGE.capitalize()} trading engine initialized successfully")
        return True
    except Exception as e:
        st.error(f"Failed to initialize {EXCHANGE} trading engine: {e}")
        logger.error(f"{EXCHANGE.capitalize()} engine initialization failed: {e}", exc_info=True)
        return False


# --- Initialize client ---
def initialize_client():
    if st.session_state.client is None:
        client_class = BinanceClient if EXCHANGE == "binance" else BybitClient
        st.session_state.client = client_class()
        if st.session_state.client._test_connection():
            logger.info(f"{EXCHANGE.capitalize()} client connected successfully")
        else:
            st.warning(f"{EXCHANGE.capitalize()} client connection failed. Check API keys.")
            logger.error(f"{EXCHANGE.capitalize()} client connection test failed")


# --- Wallet Balance ---
def get_wallet_balance() -> dict:
    mode = st.session_state.trading_modes.get(EXCHANGE, "virtual")
    cache_key = f"{mode}_{EXCHANGE}"
    default_virtual = {"capital": 1000.0, "available": 1000.0, "used": 0.0}
    default_real = {"capital": 0.0, "available": 0.0, "used": 0.0}

    if cache_key in st.session_state.wallet_cache:
        logger.info(f"Returning cached {mode} balance for {EXCHANGE}: {st.session_state.wallet_cache[cache_key]}")
        return st.session_state.wallet_cache[cache_key]

    balance_data = default_virtual if mode == "virtual" else default_real
    try:
        if mode == "virtual":
            wallet = st.session_state.engine.db.get_wallet_balance("virtual") if st.session_state.engine else None
            if wallet:
                balance_data = {
                    "capital": getattr(wallet, "capital", default_virtual["capital"]),
                    "available": getattr(wallet, "available", default_virtual["available"]),
                    "used": getattr(wallet, "used", default_virtual["used"])
                }
                logger.info(f"Fetched virtual wallet balance for {EXCHANGE}: {balance_data}")
        else:
            initialize_client()
            client = st.session_state.client
            if client and client.is_connected():
                st.session_state.engine.sync_real_balance()
                wallet = st.session_state.engine.db.get_wallet_balance("real")
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
                    logger.info(f"Fetched real wallet balance after sync for {EXCHANGE}: {balance_data}")
                else:
                    st.error(f"❌ Failed to retrieve real balance. Check {EXCHANGE} account or API permissions.")
            else:
                wallet = st.session_state.engine.db.get_wallet_balance("real") if st.session_state.engine else None
                if wallet:
                    balance_data = {
                        "capital": getattr(wallet, "capital", default_real["capital"]),
                        "available": getattr(wallet, "available", default_real["available"]),
                        "used": getattr(wallet, "used", default_real["used"])
                    }
                st.warning(f"{EXCHANGE.capitalize()} API not connected. Check API keys in .env file.")
    except Exception as e:
        logger.error(f"Error fetching {mode} wallet for {EXCHANGE}: {e}", exc_info=True)
        balance_data = default_virtual if mode == "virtual" else default_real

    st.session_state.wallet_cache[cache_key] = balance_data
    return balance_data


# --- Feedback ---
def display_feedback():
    st.header("📊 ML Feedback")
    try:
        feedback_list = db_manager.get_feedback(limit=50)
        if not feedback_list:
            st.info(f"No feedback data available for {EXCHANGE}.")
            return

        import pandas as pd
        feedback_data = [
            {
                "Symbol": f.signal.get("symbol", "N/A"),
                "Outcome": "Success" if f.outcome else "Failure",
                "Score": f.signal.get("ml_score", "N/A"),
                "Timestamp": f.timestamp,
                "Exchange": f.exchange
            }
            for f in feedback_list
        ]
        df = pd.DataFrame(feedback_data)
        st.dataframe(df, use_container_width=True)

        success_rate = sum(1 for f in feedback_list if f.outcome) / len(feedback_list) * 100 if feedback_list else 0
        st.metric("Feedback Success Rate", f"{success_rate:.2f}%")
    except Exception as e:
        st.error(f"Error loading feedback for {EXCHANGE}: {e}")
        logger.error(f"Error displaying feedback for {EXCHANGE}: {e}", exc_info=True)


# --- Main App ---
def main():
    st.markdown("""
    <style>
    .main-header { background: linear-gradient(135deg, #1e1e2f 0%, #2a2a4a 100%); padding:2rem; border-radius:10px; text-align:center; margin-bottom:2rem; border:2px solid #00ff88;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="main-header">
        <h1 style="color:#00ff88; margin:0; font-size:3rem;">🚀 AlgoTraderPro - Multi Exchange</h1>
        <p style="color:#888; margin:0; font-size:1.2rem;">Advanced Cryptocurrency Trading Platform ({EXCHANGE.capitalize()})</p>
    </div>
    """, unsafe_allow_html=True)

    if not initialize_engine():
        st.stop()

    # Load saved mode for this exchange
    if EXCHANGE not in st.session_state.trading_modes:
        saved_mode = st.session_state.engine.db.get_setting(f"trading_mode_{EXCHANGE}")
        st.session_state.trading_modes[EXCHANGE] = saved_mode if saved_mode in ["virtual", "real"] else "virtual"

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")

        # Exchange selector
        exchange_options = ["Binance", "Bybit"]
        selected_exchange = st.selectbox(
            "Exchange",
            exchange_options,
            index=0 if EXCHANGE == "binance" else 1
        )
        if selected_exchange.lower() != EXCHANGE:
            set_exchange(selected_exchange)

        # Trading mode
        mode_options = ["Virtual", "Real"]
        selected_mode = st.selectbox(
            "Trading Mode",
            mode_options,
            index=0 if st.session_state.trading_modes[EXCHANGE] == "virtual" else 1
        )
        if selected_mode.lower() != st.session_state.trading_modes[EXCHANGE]:
            st.session_state.trading_modes[EXCHANGE] = selected_mode.lower()
            st.session_state.engine.db.save_setting(f"trading_mode_{EXCHANGE}", st.session_state.trading_modes[EXCHANGE])
            st.session_state.wallet_cache.clear()
            if st.session_state.trading_modes[EXCHANGE] == "real":
                initialize_client()
                if st.session_state.client and st.session_state.client.is_connected():
                    st.session_state.engine.sync_real_balance()
            st.rerun()

        # Status
        engine_status = "🟢 Online" if st.session_state.engine_initialized else "🔴 Offline"
        st.markdown(f"**Engine Status:** {engine_status}")
        mode_color = "🟢" if st.session_state.trading_modes[EXCHANGE] == "virtual" else "🟡"
        st.markdown(f"**Trading Mode:** {mode_color} {st.session_state.trading_modes[EXCHANGE].title()}")
        st.markdown(f"**Exchange:** {EXCHANGE.capitalize()}")

        # API
        initialize_client()
        api_status = "✅ Connected" if st.session_state.client and st.session_state.client.is_connected() else "❌ Disconnected"
        st.markdown(f"**API Status:** {api_status}")

        st.divider()

        # Pages
        pages = {
            "📊 Dashboard": "pages/dashboard.py",
            "🎯 Signals": "pages/signals.py",
            "📈 Trades": "pages/trades.py",
            "📊 Performance": "pages/performance.py",
            "⚙️ Settings": "pages/settings.py",
            "📊 ML Feedback": display_feedback
        }
        for name, path in pages.items():
            if isinstance(path, str):
                if st.button(name):
                    st.switch_page(path)
            else:
                if st.button(name):
                    path()

        st.divider()

        # Wallet
        balance_data = get_wallet_balance()
        if st.session_state.trading_modes[EXCHANGE] == "virtual":
            st.metric(f"Virtual Balance ({EXCHANGE.capitalize()})", f"${balance_data['available']:.2f}")
        else:
            st.metric(f"Real Capital ({EXCHANGE.capitalize()})", f"${balance_data['capital']:.2f}")
            st.metric(f"Real Available ({EXCHANGE.capitalize()})", f"${balance_data['available']:.2f}")
            st.metric(f"Used Margin ({EXCHANGE.capitalize()})", f"${balance_data['used']:.2f}")

        st.markdown(
            f"<small style='color:#888;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>",
            unsafe_allow_html=True
        )

        if st.button("🛑 Emergency Stop"):
            st.session_state.wallet_cache.clear()
            if st.session_state.engine:
                st.session_state.engine.emergency_stop()
            st.success(f"All automated trading stopped for {EXCHANGE}")

    # Dashboard
    try:
        from pages.dashboard import main as dashboard_main
        dashboard_main()
    except Exception as e:
        st.error(f"Error loading dashboard for {EXCHANGE}: {e}")
        logger.error(f"Dashboard error for {EXCHANGE}: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:#888;'> AlgoTrader Pro - Multi Exchange (Binance & Bybit) v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exchange: {EXCHANGE.capitalize()}</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
