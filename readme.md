AlgoTrader Pro
AlgoTrader Pro is a cryptocurrency algorithmic trading platform built with Streamlit, supporting automated trading on both Binance and Bybit exchanges. It operates in virtual (paper trading) and real trading modes, offering signal generation with technical indicators, machine learning filtering, portfolio management, ML feedback storage, and comprehensive performance analytics.
The platform scans multiple cryptocurrency markets, generates trading signals using technical analysis, executes trades automatically, and provides real-time monitoring and reporting through an intuitive web interface.

System Architecture
Frontend Architecture

Streamlit serves as the primary web framework with a multi-page architecture.
Main entry point (app.py) initializes the trading engine, manages session state, and supports dynamic exchange selection (Binance or Bybit).
Individual pages in the pages/ directory include Dashboard, Signals, Trades, Performance Analytics, Settings, and ML Feedback.

Backend Architecture

TradingEngine (engine.py for Bybit, binance_trading_engine.py for Binance) orchestrates trading operations, dynamically loaded based on the EXCHANGE environment variable.
Exchange Clients (binance_client.py, bybit_client.py) handle API communication (REST + WebSocket) for their respective exchanges.
AutomatedTrader (automated_trader.py) manages trading loops and execution.
SignalGenerator (signal_generator.py) creates technical analysis signals.
MLFilter (ml.py) applies machine learning to filter and score trading signals.
Indicators (indicators.py) fetches market data and calculates technical indicators for both exchanges.

Data Storage

PostgreSQL with SQLAlchemy ORM stores trades, signals, wallet balances, settings, and ML feedback.
Models: SignalModel, TradeModel, WalletBalanceModel, SettingsModel, FeedbackModel.


Alembic manages database migrations.
JSON config files: settings.json, capital.json, virtual_trades.json.

Signal Generation

Indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Volume.
ML filtering with XGBoost (ml.py) using features: rsi, macd, macd_signal, macd_histogram, bb_position, volume_ratio, trend_score, volatility, price_change_1h, price_change_4h, price_change_24h.
Feedback storage in FeedbackModel for ML model retraining.

Risk Management

Position sizing based on account balance.
Stop-loss and take-profit automation.
Drawdown limits, leverage controls.
Virtual trading mode for safe testing.


External Dependencies

Binance/Bybit APIs – market data and trade execution.
pandas, numpy – data wrangling.
plotly – interactive charts.
scikit-learn, xgboost – ML filtering.
sqlalchemy, alembic, psycopg2-binary – database and migrations.
requests, tenacity – API handling.
streamlit – web UI.
discord.py, telegram-bot, WhatsApp integration – notifications.


🚀 Setup Guide
1. Clone Repository
git clone https://github.com/yourusername/algotrader-pro.git
cd algotrader-pro

2. Environment Variables
Create a .env file:
EXCHANGE=binance  # or 'bybit'
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
DB_URL=postgresql+psycopg2://trader:securepass@db:5432/algotrader


🐳 Dockerized Setup (Streamlit + PostgreSQL + pgAdmin4)
docker-compose.yml
version: "3.9"

services:
  db:
    image: postgres:15
    container_name: algotrader-db
    restart: always
    environment:
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: securepass
      POSTGRES_DB: algotrader
    ports:
      - "5432:5432"
    volumes:
      - db_data:/var/lib/postgresql/data

  pgadmin:
    image: dpage/pgadmin4
    container_name: algotrader-pgadmin
    restart: always
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@algotrader.local
      PGADMIN_DEFAULT_PASSWORD: adminpass
    ports:
      - "5050:80"
    depends_on:
      - db

  app:
    build: .
    container_name: algotrader-app
    restart: always
    environment:
      EXCHANGE: ${EXCHANGE:-binance}
      BINANCE_API_KEY: ${BINANCE_API_KEY}
      BINANCE_API_SECRET: ${BINANCE_API_SECRET}
      BYBIT_API_KEY: ${BYBIT_API_KEY}
      BYBIT_API_SECRET: ${BYBIT_API_SECRET}
      DB_URL: postgresql+psycopg2://trader:securepass@db:5432/algotrader
    volumes:
      - .:/app
    ports:
      - "8501:8501"
    depends_on:
      - db
    command: streamlit run app.py --server.port=8501 --server.address=0.0.0.0

volumes:
  db_data:

Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

requirements.txt
streamlit
sqlalchemy
psycopg2-binary
pandas
numpy
plotly
scikit-learn
xgboost
requests
python-dotenv
alembic
tenacity


🛠 Database Migrations with Alembic
Initialize Alembic
alembic init migrations

This creates a migrations/ folder and alembic.ini.
Update alembic.ini
Set SQLAlchemy URL:
sqlalchemy.url = postgresql+psycopg2://trader:securepass@db:5432/algotrader

Generate Migration
alembic revision --autogenerate -m "create initial tables including feedback"

Apply Migration
alembic upgrade head


🔗 Access

Streamlit App → http://localhost:8501
pgAdmin4 → http://localhost:5050
Login: admin@algotrader.local / adminpass
Add Server → Host: db, User: trader, Password: securepass




✅ Development Flow

Edit code/models in Python (e.g., db.py, ml.py, indicators.py).

Run alembic revision --autogenerate -m "update tables" when models change (e.g., adding FeedbackModel).

Run alembic upgrade head to apply DB schema changes.

Rebuild containers:
docker-compose up --build




📊 Features

Multi-exchange support (Binance, Bybit) via EXCHANGE environment variable.
Technical indicator-based signal generation (SMA, EMA, RSI, MACD, Bollinger Bands, ATR).
ML-driven signal filtering with XGBoost, using features like RSI, MACD, Bollinger Band position, and price changes.
ML feedback storage and management via FeedbackModel in PostgreSQL.
Real-time portfolio monitoring and performance analytics.
Virtual and real trading modes.
Risk management with stop-loss, take-profit, and leverage controls.
Web-based UI with Streamlit for dashboard, signals, trades, performance, settings, and feedback.
Notifications via Discord, Telegram, and WhatsApp.

Features in Progress

Strategy backtesting module.
Portfolio rebalancing.
Advanced ML models (e.g., deep learning).
Additional exchange integrations.