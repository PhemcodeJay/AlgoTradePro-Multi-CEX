# AlgoTrader Pro - Replit Configuration

## Project Overview
AlgoTrader Pro is a cryptocurrency algorithmic trading platform built with Streamlit. It supports automated trading on Binance and Bybit exchanges with both virtual (paper trading) and real trading modes.

## Key Features
- Multi-exchange support (Binance and Bybit)
- Technical indicator-based signal generation (SMA, EMA, RSI, MACD, Bollinger Bands, ATR)
- ML-driven signal filtering with XGBoost
- Real-time portfolio monitoring and performance analytics
- Virtual and real trading modes
- Risk management with stop-loss, take-profit, and leverage controls
- Web-based UI with Streamlit

## Tech Stack
- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python 3.11
- **Database**: PostgreSQL (Replit-managed)
- **ML**: XGBoost, scikit-learn
- **APIs**: CCXT for exchange integration
- **Deployment**: Runs on port 5000

## Environment Setup

### Required Environment Variables
The following are automatically configured by Replit:
- `DATABASE_URL` - PostgreSQL connection string
- `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`, `PGPORT` - PostgreSQL credentials

### Optional API Keys (for real trading)
Add these in Replit Secrets for real trading mode:
- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_API_SECRET` - Your Binance API secret
- `BYBIT_API_KEY` - Your Bybit API key
- `BYBIT_API_SECRET` - Your Bybit API secret

### Optional Notification Services
- `DISCORD_WEBHOOK_URL` - For Discord notifications
- `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` - For Telegram notifications

## Project Structure
```
├── app.py                      # Main Streamlit application
├── pages/                      # Streamlit pages
│   ├── 1_Dashboard.py
│   ├── 2_Signals.py
│   ├── 3_Trades.py
│   ├── 4_Performance.py
│   ├── 5_Settings.py
│   └── 6_ML_Feedback.py
├── db.py                       # Database models and manager
├── multi_trading_engine.py     # Trading engine orchestration
├── binance_client.py           # Binance API client
├── bybit_client.py             # Bybit API client
├── signal_generator.py         # Signal generation logic
├── ml.py                       # Machine learning filter
├── indicators.py               # Technical indicators
└── automated_trader.py         # Automated trading logic
```

## Running the Application

### Development Mode
The application runs automatically via the configured workflow on port 5000.

### Virtual Trading Mode (Default)
- No API keys required
- Uses simulated trades and market data
- Perfect for testing strategies

### Real Trading Mode
- Requires exchange API credentials in Secrets
- Executes actual trades on the exchange
- Use with caution and proper risk management

## Database
The application uses PostgreSQL for storing:
- Trading signals
- Trade history
- Wallet balances
- Settings
- ML feedback data

Tables are automatically created on first run.

## Recent Changes (October 2025)
- ✅ Imported from GitHub
- ✅ Fixed Python dependency conflicts (tenacity version)
- ✅ Fixed f-string syntax error in binance_client.py
- ✅ Configured for Replit environment (port 5000)
- ✅ Streamlit proxy configuration for Replit iframe
- ✅ PostgreSQL database integrated
- ✅ All dependencies installed via uv
- ✅ Application tested and verified working

## Notes
- The application defaults to Bybit exchange in virtual mode
- Exchange can be switched via the UI
- Trading mode can be toggled between Virtual and Real (when API keys are present)
