Got it 👍 — you want that Markdown documentation cleaned up, formatted properly, and free of repetition, syntax issues, and layout inconsistencies.
Here’s a **fully fixed and polished version** of your `README.md` for **AlgoTrader Pro**:

---

# 🧠 AlgoTrader Pro

**AlgoTrader Pro** is a cryptocurrency algorithmic trading platform built with **Streamlit**, supporting automated trading on **Binance** and **Bybit**.
It operates in **virtual (paper trading)** and **real trading** modes, featuring signal generation, machine learning filtering, portfolio management, ML feedback storage, and performance analytics.

The system continuously scans markets, generates trading signals using technical indicators, executes trades automatically, and provides real-time monitoring through a web interface.

---

## 🚀 Key Features

* **Multi-Exchange Support**: Trade on Binance or Bybit (set via `EXCHANGE` variable or Streamlit UI).
* **Signal Generation**: Uses indicators such as SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and Volume.
* **ML-Driven Filtering**: XGBoost-based signal scoring with RSI, MACD, Bollinger Band position, and price change features.
* **Portfolio Monitoring**: Real-time trade, wallet balance, and analytics tracking.
* **Trading Modes**:

  * **Virtual (default)** – Simulated trading for testing.
  * **Real** – Live trading with API keys.
* **Risk Management**: Stop-loss, take-profit, leverage, and drawdown limit controls.
* **Notifications**: Integrated with Discord, Telegram, and WhatsApp.
* **Database**: PostgreSQL with SQLAlchemy for persistent storage (trades, signals, balances, feedback).

---

## 🧩 System Architecture

### **Frontend**

* **Framework**: Streamlit (multi-page structure)
* **Main Entry Point**: `app.py` – initializes the trading engine and manages session state
* **Pages** (`pages/` directory):

  * `1_Dashboard.py`
  * `2_Signals.py`
  * `3_Trades.py`
  * `4_Performance.py`
  * `5_Settings.py`
  * `6_ML_Feedback.py`

---

### **Backend**

| Module                                  | Description                                              |
| --------------------------------------- | -------------------------------------------------------- |
| `multi_trading_engine.py`               | Main orchestrator for trading engines (Binance or Bybit) |
| `binance_client.py` / `bybit_client.py` | Exchange API clients using REST & WebSocket              |
| `automated_trader.py`                   | Handles continuous trading loops                         |
| `signal_generator.py`                   | Generates buy/sell signals based on TA                   |
| `ml.py`                                 | ML filtering using XGBoost                               |
| `indicators.py`                         | Calculates technical indicators                          |

---

### **Data Storage**

* **PostgreSQL + SQLAlchemy ORM**

  * `SignalModel` – Trading signals
  * `TradeModel` – Trade history
  * `WalletBalanceModel` – Account balances
  * `SettingsModel` – Configurations
  * `FeedbackModel` – ML feedback
* **Alembic** – Database migrations
* **JSON Files** – Legacy storage (`settings.json`, `capital.json`, `virtual_trades.json`)

---

## 🧱 Project Structure

```
AlgoTraderPro/
├── app.py                      # Main Streamlit application
├── pages/                      # Streamlit pages
│   ├── 1_Dashboard.py
│   ├── 2_Signals.py
│   ├── 3_Trades.py
│   ├── 4_Performance.py
│   ├── 5_Settings.py
│   └── 6_ML_Feedback.py
├── db.py                       # Database models and manager
├── multi_trading_engine.py     # Trading engine orchestrator
├── binance_client.py           # Binance API client
├── bybit_client.py             # Bybit API client
├── signal_generator.py         # Signal generation logic
├── ml.py                       # Machine learning filter
├── indicators.py               # Technical indicators
├── automated_trader.py         # Automated trading logic
├── requirements.txt            # Dependencies
├── .env                        # Environment variables (excluded from Git)
├── migrations/                 # Alembic migrations
└── alembic.ini                 # Alembic configuration
```

---

## ⚙️ Setup Guide (Local Development)

### **Prerequisites**

* Python 3.11+
* PostgreSQL (local or AWS RDS)
* Git
* OS: Windows (Git Bash), macOS, or Linux

---

### **1. Clone the Repository**

```bash
git clone https://github.com/PhemcodeJay/AlgoTradePro-Multi-CEX.git
cd AlgoTradePro-Multi-CEX
```

---

### **2. Set Up Python Environment**

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
# or
source venv/Scripts/activate  # Windows (Git Bash)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **3. Create a `.env` File**

Create `.env` in the project root:

```bash
nano .env
```

**Example:**

```
EXCHANGE=binance  # or 'bybit'
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret

DATABASE_URL=postgresql+psycopg2://postgres:kokochulo1234@algotrader-db.ctuauq84wu6f.eu-north-1.rds.amazonaws.com:5432/algotrader-db

DISCORD_WEBHOOK_URL=your_discord_webhook
TELEGRAM_BOT_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id
WHATSAPP_TO=your_phone_number
```

For **local PostgreSQL**:

```
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/algotrader
```

Secure the file:

```bash
chmod 600 .env
```

---

### **4. Set Up PostgreSQL**

#### **Option A – AWS RDS**

Make sure your RDS allows connections on port `5432`.
Test connection:

```bash
PGPASSWORD=kokochulo1234 psql -h algotrader-db.ctuauq84wu6f.eu-north-1.rds.amazonaws.com -U postgres -d algotrader-db -c "\q"
```

#### **Option B – Local PostgreSQL**

Install and create the database:

```bash
sudo apt install postgresql postgresql-contrib
psql -U postgres -c "CREATE DATABASE algotrader;"
```

Initialize tables:

```bash
python3 -c "from db import db_manager; db_manager.create_tables()"
```

---

### **5. Initialize Alembic for Migrations**

```bash
alembic init migrations
```

Edit `alembic.ini`:

```
sqlalchemy.url = postgresql+psycopg2://postgres:kokochulo1234@algotrader-db.ctuauq84wu6f.eu-north-1.rds.amazonaws.com:5432/algotrader-db
```

Generate and apply migrations:

```bash
alembic revision --autogenerate -m "create initial tables"
alembic upgrade head
```

---

### **6. Run the Streamlit App**

```bash
streamlit run app.py --server.port=5000 --server.address=0.0.0.0
```

Access the app at:
👉 [http://localhost:5000](http://localhost:5000)

---

## 🧪 Running Modes

### **Virtual Trading Mode (Default)**

* No API keys required
* Simulated trades & price data
* Perfect for testing and debugging

### **Real Trading Mode**

* Requires API credentials (`.env`)
* Executes live trades on selected exchange
  ⚠️ Use with caution and proper risk management.

---

## 🗄️ Database Overview

PostgreSQL stores:

* Trading signals
* Trade history
* Wallet balances
* Settings
* ML feedback

Tables are automatically created on first run via `db.py`.

---