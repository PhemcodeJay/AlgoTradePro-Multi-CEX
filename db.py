import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Union
from sqlalchemy import Index, create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from logging_config import get_trading_logger
from contextlib import contextmanager

from dotenv import load_dotenv
load_dotenv()  # This loads .env into os.environ

logger = get_trading_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required but not set")
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False, pool_size=10, max_overflow=20, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
ScopedSession = scoped_session(SessionLocal)  # Thread-safe session factory

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(64), nullable=False)
    binance_api_key = Column(String(100), nullable=True)
    binance_api_secret = Column(String(100), nullable=True)
    bybit_api_key = Column(String(100), nullable=True)
    bybit_api_secret = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "binance_api_key": self.binance_api_key,
            "binance_api_secret": self.binance_api_secret,
            "bybit_api_key": self.bybit_api_key,
            "bybit_api_secret": self.bybit_api_secret,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    interval = Column(String(10), nullable=False)
    signal_type = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    score = Column(Float, nullable=False)
    entry = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    tp = Column(Float, nullable=False)
    trail = Column(Float, nullable=False, default=0.0)
    liquidation = Column(Float, nullable=False, default=0.0)
    leverage = Column(Integer, nullable=False)
    margin_usdt = Column(Float, nullable=False, default=0.0)
    market = Column(String(50), nullable=False)
    indicators = Column(JSON, nullable=True)
    exchange = Column(String(20), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    __table_args__ = (Index('ix_signals_created_at', 'created_at'),)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'interval': self.interval,
            'signal_type': self.signal_type,
            'side': self.side,
            'score': self.score,
            'entry': self.entry,
            'sl': self.sl,
            'tp': self.tp,
            'trail': self.trail,
            'liquidation': self.liquidation,
            'leverage': self.leverage,
            'margin_usdt': self.margin_usdt,
            'market': self.market,
            'indicators': self.indicators,
            'exchange': self.exchange,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None
        }

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    status = Column(String(20), default="open", index=True)
    virtual = Column(Boolean, default=True, nullable=False)
    leverage = Column(Integer, default=10)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    trail = Column(Float, nullable=True)
    margin_usdt = Column(Float, nullable=True)
    exchange = Column(String(20), nullable=True, index=True)
    signal_id = Column(Integer, nullable=True)
    order_id = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    indicators = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'side': self.side,
            'qty': self.qty,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'status': self.status,
            'virtual': self.virtual,
            'leverage': self.leverage,
            'sl': self.sl,
            'tp': self.tp,
            'trail': self.trail,
            'margin_usdt': self.margin_usdt,
            'exchange': self.exchange,
            'signal_id': self.signal_id,
            'order_id': self.order_id,
            'error_message': self.error_message,
            'indicators': self.indicators,
            'created_at': self.created_at.isoformat() if self.created_at is not None else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None
        }

class WalletBalance(Base):
    __tablename__ = "wallet_balances"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    account_type = Column(String(20), nullable=False, index=True)
    available = Column(Float, nullable=False, default=0.0)
    used = Column(Float, nullable=False, default=0.0)
    total = Column(Float, nullable=False, default=0.0)
    currency = Column(String(10), default="USDT")
    exchange = Column(String(20), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'account_type': self.account_type,
            'available': self.available,
            'used': self.used,
            'total': self.total,
            'currency': self.currency,
            'exchange': self.exchange,
            'updated_at': self.updated_at.isoformat() if self.updated_at is not None else None
        }

class Settings(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

class FeedbackModel(Base):
    __tablename__ = "ml_feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    outcome = Column(Boolean, nullable=False)
    pnl = Column(Float, nullable=True)
    indicators = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timestamp': self.timestamp.isoformat() if self.timestamp is not None else None,
            'outcome': self.outcome,
            'pnl': self.pnl,
            'indicators': self.indicators
        }

class ErrorLog(Base):
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    error_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    error_code = Column(String(50), nullable=True)
    context = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    trading_mode = Column(String(20), nullable=False, default=TRADING_MODE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'error_type': self.error_type,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat() if self.timestamp is not None else None,
            'trading_mode': self.trading_mode
        }

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600
        )
        self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))

    @contextmanager
    def get_session(self):
        """Provide a thread-safe session"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    def add_user(self, user_data: Dict[str, Any]) -> bool:
        with self.get_session() as session:
            try:
                existing_user = session.query(User).filter(User.username == user_data["username"]).first()
                if existing_user:
                    logger.error(f"Username {user_data['username']} already exists")
                    return False
                user = User(**user_data)
                session.add(user)
                session.commit()
                # Initialize wallet balances for new user
                session.add(WalletBalance(
                    user_id=user.id,
                    account_type="virtual",
                    available=100.0,
                    used=0.0,
                    total=100.0,
                    currency="USDT",
                    exchange="binance"
                ))
                session.add(WalletBalance(
                    user_id=user.id,
                    account_type="real",
                    available=0.0,
                    used=0.0,
                    total=0.0,
                    currency="USDT",
                    exchange="binance"
                ))
                session.commit()
                logger.info(f"User {user_data['username']} added with wallet balances")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding user: {e}")
                session.rollback()
                return False

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            try:
                user = session.query(User).filter(User.username == username).first()
                return user.to_dict() if user else None
            except SQLAlchemyError as e:
                logger.error(f"Error fetching user {username}: {e}")
                return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            try:
                user = session.query(User).filter(User.id == user_id).first()
                return user.to_dict() if user else None
            except SQLAlchemyError as e:
                logger.error(f"Error fetching user by ID {user_id}: {e}")
                return None

    def add_signal(self, signal: Dict[str, Any]) -> bool:
        with self.get_session() as session:
            try:
                signal_model = Signal(**signal)
                session.add(signal_model)
                session.commit()
                logger.debug(f"Signal added for {signal['symbol']}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding signal: {e}")
                return False
    
    def get_signals(self, limit: int = 100, exchange: Optional[str] = None, user_id: Optional[int] = None, symbol: Optional[str] = None, side: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            try:
                query = session.query(Signal)
                if exchange:
                    query = query.filter(Signal.exchange == exchange)
                if user_id:
                    query = query.filter(Signal.user_id == user_id)
                if symbol:
                    query = query.filter(Signal.symbol == symbol)
                if side:
                    query = query.filter(Signal.side == side)
                signals = query.order_by(Signal.created_at.desc()).limit(limit).all()
                return [signal.to_dict() for signal in signals]
            except SQLAlchemyError as e:
                logger.error(f"Error fetching signals: {e}")
                return []
            
    def add_trade(self, trade: Dict[str, Any]) -> bool:
        with self.get_session() as session:
            try:
                trade_model = Trade(**trade)
                session.add(trade_model)
                session.commit()
                logger.debug(f"Trade added for {trade['symbol']}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding trade: {e}")
                return False

    def get_trades(self, limit: int = 100, exchange: Optional[str] = None, virtual: Optional[bool] = None, user_id: Optional[int] = None, hours: Optional[int] = None, status: Optional[str] = None) -> List[Trade]:
        with self.get_session() as session:
            try:
                query = session.query(Trade)
                if exchange:
                    query = query.filter(Trade.exchange == exchange)
                if virtual is not None:
                    query = query.filter(Trade.virtual == virtual)
                if user_id:
                    query = query.filter(Trade.user_id == user_id)
                if hours:
                    since = datetime.now(timezone.utc) - timedelta(hours=hours)
                    query = query.filter(Trade.updated_at >= since)
                if status:
                    query = query.filter(Trade.status == status)
                return query.order_by(Trade.updated_at.desc()).limit(limit).all()
            except SQLAlchemyError as e:
                logger.error(f"Error fetching trades: {e}")
                return []

    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> bool:
        with self.get_session() as session:
            try:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if not trade:
                    logger.warning(f"No trade found for ID {trade_id}")
                    return False
                for key, value in updates.items():
                    setattr(trade, key, value)
                session.commit()
                logger.debug(f"Trade {trade_id} updated")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error updating trade {trade_id}: {e}")
                return False

    def get_wallet_balance(self, account_type: str, user_id: int, exchange: Optional[str] = None) -> Optional[Dict[str, float]]:
        with self.get_session() as session:
            try:
                query = session.query(WalletBalance).filter(
                    WalletBalance.account_type == account_type,
                    WalletBalance.user_id == user_id
                )
                if exchange:
                    query = query.filter(WalletBalance.exchange == exchange)
                balance = query.first()
                if not balance:
                    # Initialize default balance if none exists
                    available = 100.0 if account_type == "virtual" else 0.0
                    balance = WalletBalance(
                        user_id=user_id,
                        account_type=account_type,
                        available=available,
                        used=0.0,
                        total=available,
                        currency="USDT",
                        exchange=exchange
                    )
                    session.add(balance)
                    session.commit()
                    logger.info(f"Initialized {account_type} wallet balance for user {user_id}, exchange {exchange}: ${available}")
                return balance.to_dict() if balance else {"available": available, "used": 0.0, "total": available}
            except SQLAlchemyError as e:
                logger.error(f"Error fetching wallet balance for {account_type}, user {user_id}, exchange {exchange}: {e}")
                return {"available": 100.0 if account_type == "virtual" else 0.0, "used": 0.0, "total": 100.0 if account_type == "virtual" else 0.0}

    def update_wallet_balance(self, account_type: str, user_id: int, available: float, used: float, exchange: Optional[str] = None) -> bool:
        with self.get_session() as session:
            try:
                query = session.query(WalletBalance).filter(
                    WalletBalance.account_type == account_type,
                    WalletBalance.user_id == user_id
                )
                if exchange:
                    query = query.filter(WalletBalance.exchange == exchange)
                balance = query.first()
                if not balance:
                    balance = WalletBalance(
                        user_id=user_id,
                        account_type=account_type,
                        available=available,
                        used=used,
                        total=available + used,
                        currency="USDT",
                        exchange=exchange
                    )
                    session.add(balance)
                else:
                    balance.available = available
                    balance.used = used
                    balance.total = available + used
                session.commit()
                logger.debug(f"Wallet balance updated: {account_type} for user {user_id}, exchange {exchange}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error updating wallet balance for {account_type}, user {user_id}, exchange {exchange}: {e}")
                session.rollback()
                return False

    def add_feedback(self, feedback: Union[FeedbackModel, Dict[str, Any]]) -> bool:
        with self.get_session() as session:
            try:
                if isinstance(feedback, dict):
                    feedback_model = FeedbackModel(**feedback)
                else:
                    feedback_model = feedback
                session.add(feedback_model)
                session.commit()
                logger.debug(f"Feedback added for {feedback_model.symbol}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding feedback: {e}")
                session.rollback()
                return False

    def get_feedback(self, limit: int = 100, exchange: Optional[str] = None, user_id: Optional[int] = None) -> List[FeedbackModel]:
        with self.get_session() as session:
            try:
                query = session.query(FeedbackModel)
                if exchange:
                    query = query.filter(FeedbackModel.exchange == exchange)
                if user_id:
                    query = query.filter(FeedbackModel.user_id == user_id)
                return query.order_by(FeedbackModel.timestamp.desc()).limit(limit).all()
            except SQLAlchemyError as e:
                logger.error(f"Error fetching feedback: {e}")
                return []

    def add_error_log(self, error_log: Dict[str, Any]) -> bool:
        with self.get_session() as session:
            try:
                error_model = ErrorLog(**error_log)
                session.add(error_model)
                session.commit()
                logger.debug(f"Error log added: {error_log['error_type']} - {error_log['message']}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding error log: {e}")
                return False

    def get_trade(self, order_id: str) -> Optional[Trade]:
        with self.get_session() as session:
            try:
                trade = session.query(Trade).filter(Trade.order_id == order_id).first()
                if trade:
                    logger.debug(f"Trade retrieved for order_id: {order_id}")
                else:
                    logger.warning(f"No trade found for order_id: {order_id}")
                return trade
            except SQLAlchemyError as e:
                logger.error(f"Error fetching trade for order_id {order_id}: {e}")
                return None

    def get_trade_by_position(self, symbol: str, exchange: str, virtual: bool, user_id: int) -> Optional[Trade]:
        with self.get_session() as session:
            try:
                trade = session.query(Trade).filter(
                    Trade.symbol == symbol,
                    Trade.exchange == exchange,
                    Trade.virtual == virtual,
                    Trade.status == "open",
                    Trade.user_id == user_id
                ).first()
                if trade:
                    logger.debug(f"Trade retrieved for position: {symbol} - {exchange} - virtual: {virtual}")
                else:
                    logger.warning(f"No open trade found for position: {symbol} - {exchange} - virtual: {virtual}")
                return trade
            except SQLAlchemyError as e:
                logger.error(f"Error fetching trade for position {symbol} - {exchange}: {e}")
                return None

    def migrate_capital_json_to_db(self, capital_file: str = "capital.json", user_id: int = 1):
        with self.get_session() as session:
            try:
                if os.path.exists(capital_file):
                    with open(capital_file, 'r') as f:
                        capital_data = json.load(f)

                    if 'virtual' in capital_data:
                        virtual_balance = capital_data['virtual']
                        self.update_wallet_balance(
                            account_type="virtual",
                            user_id=user_id,
                            available=virtual_balance.get("available", 100.0),
                            used=virtual_balance.get("used", 0.0),
                            exchange=virtual_balance.get("exchange", "binance")
                        )

                    if 'real' in capital_data:
                        real_balance = capital_data['real']
                        self.update_wallet_balance(
                            account_type="real",
                            user_id=user_id,
                            available=real_balance.get("available", 0.0),
                            used=real_balance.get("used", 0.0),
                            exchange=real_balance.get("exchange", "binance")
                        )

                    logger.info(f"Capital data migrated to database for user {user_id}")
                    # Optionally, rename or delete the capital.json file to prevent re-migration
                    os.rename(capital_file, f"{capital_file}.bak")
                else:
                    # Initialize default balances if no capital.json exists
                    self.update_wallet_balance(
                        account_type="virtual",
                        user_id=user_id,
                        available=100.0,
                        used=0.0,
                        exchange="binance"
                    )
                    self.update_wallet_balance(
                        account_type="real",
                        user_id=user_id,
                        available=0.0,
                        used=0.0,
                        exchange="binance"
                    )
                    logger.info(f"Default wallet balances created for user {user_id}")

            except Exception as e:
                logger.error(f"Error migrating capital data for user {user_id}: {e}")
                session.rollback()
                raise e

# Global database manager instance
db_manager = DatabaseManager()

# Initialize database on import
try:
    db_manager.create_tables()
    db_manager.migrate_capital_json_to_db()
except Exception as e:
    logger.error(f"Database initialization failed: {e}")