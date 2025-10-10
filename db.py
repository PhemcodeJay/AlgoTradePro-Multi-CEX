import os
import json
from datetime import datetime, timezone
from signal_generator import Signal
from typing import List, Optional, Dict, Any, Union
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import SQLAlchemyError
from logging_config import get_trading_logger
from contextlib import contextmanager

logger = get_trading_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:1234@localhost:5432/Algotrader2")
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False, pool_size=10, max_overflow=20, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
ScopedSession = scoped_session(SessionLocal)  # Thread-safe session factory

# Models
class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
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
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))

class FeedbackModel(Base):
    __tablename__ = "ml_feedback"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    outcome = Column(Boolean, nullable=False)
    pnl = Column(Float, nullable=True)
    indicators = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
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
    error_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    error_code = Column(String(50), nullable=True)
    context = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    trading_mode = Column(String(20), nullable=False, default=TRADING_MODE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
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

    def add_signal(self, signal: Union[Signal, Dict[str, Any]]) -> bool:
        with self.get_session() as session:
            try:
                if isinstance(signal, dict):
                    signal_obj = Signal(
                        symbol=signal['symbol'],
                        interval=signal['interval'],
                        signal_type=signal['signal_type'],
                        side=signal['side'],
                        score=signal['score'],
                        entry=signal['entry'],
                        sl=signal['sl'],
                        tp=signal['tp'],
                        trail=signal.get('trail', 0.0),
                        liquidation=signal.get('liquidation', 0.0),
                        leverage=signal['leverage'],
                        margin_usdt=signal.get('margin_usdt', 0.0),
                        market=signal['market'],
                        indicators=signal['indicators'],
                        exchange=signal['exchange'],
                        created_at=datetime.now(timezone.utc)
                    )
                else:
                    signal_obj = signal
                session.add(signal_obj)
                session.commit()
                logger.debug(f"Signal added for {signal_obj.symbol}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding signal for {signal_obj.symbol}: {e}")
                return False

    def get_signals(self, limit: int = 10, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            try:
                query = session.query(Signal).order_by(Signal.created_at.desc())
                if exchange:
                    query = query.filter(Signal.exchange == exchange)
                signals = query.limit(limit).all()
                # Convert to dicts within the session
                return [signal.to_dict() for signal in signals]
            except SQLAlchemyError as e:
                logger.error(f"Error fetching signals: {e}")
                return []

    def add_trade(self, trade: Trade) -> bool:
        with self.get_session() as session:
            try:
                session.add(trade)
                session.commit()
                logger.debug(f"Trade added for {trade.symbol}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding trade: {e}")
                return False

    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> bool:
        with self.get_session() as session:
            try:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if trade:
                    for key, value in updates.items():
                        if hasattr(trade, key):
                            setattr(trade, key, value)
                    trade.updated_at = datetime.now(timezone.utc)  # type: ignore
                    session.commit()
                    logger.debug(f"Trade {trade_id} updated")
                    return True
                return False
            except SQLAlchemyError as e:
                logger.error(f"Error updating trade: {e}")
                return False

    def get_trades(self, limit: int = 100, symbol: Optional[str] = None, status: Optional[str] = None, virtual: Optional[bool] = None, exchange: Optional[str] = None) -> List[Trade]:
        with self.get_session() as session:
            try:
                query = session.query(Trade)
                if symbol:
                    query = query.filter(Trade.symbol == symbol)
                if status:
                    query = query.filter(Trade.status == status)
                if virtual is not None:
                    query = query.filter(Trade.virtual == virtual)
                if exchange:
                    query = query.filter(Trade.exchange == exchange)
                return query.order_by(Trade.created_at.desc()).limit(limit).all()
            except SQLAlchemyError as e:
                logger.error(f"Error fetching trades: {e}")
                return []

    def get_trade_by_id(self, trade_id: int) -> Optional[Trade]:
        with self.get_session() as session:
            try:
                return session.query(Trade).filter(Trade.id == trade_id).first()
            except SQLAlchemyError as e:
                logger.error(f"Error fetching trade {trade_id}: {e}")
                return None

    def get_wallet_balance(self, account_type: str, exchange: Optional[str] = None) -> Optional[WalletBalance]:
        with self.get_session() as session:
            try:
                query = session.query(WalletBalance).filter(WalletBalance.account_type == account_type)
                if exchange:
                    query = query.filter(WalletBalance.exchange == exchange)
                return query.first()
            except SQLAlchemyError as e:
                logger.error(f"Error fetching wallet balance: {e}")
                return None

    def update_wallet_balance(self, account_type: str, available: float, used: float = 0.0, exchange: Optional[str] = None) -> bool:
        with self.get_session() as session:
            try:
                wallet = session.query(WalletBalance).filter(WalletBalance.account_type == account_type).first()
                if wallet:
                    wallet.available = available  # type: ignore
                    wallet.used = used  # type: ignore
                    wallet.total = available + used  # type: ignore
                    if exchange:
                        wallet.exchange = exchange  # type: ignore
                    wallet.updated_at = datetime.now(timezone.utc)  # type: ignore
                else:
                    wallet = WalletBalance(
                        account_type=account_type,
                        available=available,
                        used=used,
                        total=available + used,
                        exchange=exchange
                    )
                    session.add(wallet)
                session.commit()
                logger.debug(f"Wallet balance updated for {account_type}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error updating wallet balance: {e}")
                return False

    def add_feedback(self, feedback: FeedbackModel) -> bool:
        with self.get_session() as session:
            try:
                session.add(feedback)
                session.commit()
                logger.debug(f"Feedback added for {feedback.symbol}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Error adding feedback: {e}")
                return False

    def get_feedback(self, limit: int = 100, exchange: Optional[str] = None) -> List[FeedbackModel]:
        with self.get_session() as session:
            try:
                query = session.query(FeedbackModel)
                if exchange:
                    query = query.filter(FeedbackModel.exchange == exchange)
                return query.order_by(FeedbackModel.timestamp.desc()).limit(limit).all()
            except SQLAlchemyError as e:
                logger.error(f"Error fetching feedback: {e}")
                return []

    def add_error_log(self, error_log: ErrorLog) -> bool:
        with self.get_session() as session:
            try:
                session.add(error_log)
                session.commit()
                logger.debug(f"Error log added: {error_log.error_type} - {error_log.message}")
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

    def get_trade_by_position(self, symbol: str, exchange: str, virtual: bool) -> Optional[Trade]:
        with self.get_session() as session:
            try:
                trade = session.query(Trade).filter(
                    Trade.symbol == symbol,
                    Trade.exchange == exchange,
                    Trade.virtual == virtual,
                    Trade.status == "open"
                ).first()
                if trade:
                    logger.debug(f"Trade retrieved for position: {symbol} - {exchange} - virtual: {virtual}")
                else:
                    logger.warning(f"No open trade found for position: {symbol} - {exchange} - virtual: {virtual}")
                return trade
            except SQLAlchemyError as e:
                logger.error(f"Error fetching trade for position {symbol} - {exchange}: {e}")
                return None

    def migrate_capital_json_to_db(self, capital_file: str = "capital.json"):
        with self.get_session() as session:
            try:
                if os.path.exists(capital_file):
                    with open(capital_file, 'r') as f:
                        capital_data = json.load(f)

                    if 'virtual' in capital_data:
                        virtual_balance = capital_data['virtual']
                        self.update_wallet_balance(
                            "virtual",
                            available=virtual_balance.get("available", 100.0),
                            used=virtual_balance.get("used", 0.0)
                        )

                    if 'real' in capital_data:
                        real_balance = capital_data['real']
                        self.update_wallet_balance(
                            "real",
                            available=real_balance.get("available", 0.0),
                            used=real_balance.get("used", 0.0)
                        )

                    logger.info("Capital data migrated to database")
                else:
                    self.update_wallet_balance("virtual", available=100.0, used=0.0)
                    logger.info("Default virtual wallet balance created")

            except Exception as e:
                logger.error(f"Error migrating capital data: {e}")

# Global database manager instance
db_manager = DatabaseManager()

# Initialize database on import
try:
    db_manager.create_tables()
    db_manager.migrate_capital_json_to_db()
except Exception as e:
    logger.error(f"Database initialization failed: {e}")