import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from logging import LogRecord

# Custom LogRecord to support extra_data
class CustomLogRecord(logging.LogRecord):
    extra_data: Optional[Dict[str, Any]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_data = kwargs.get('extra', {}).get('extra_data')

# Configure logging format
class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record: CustomLogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data') and record.extra_data:
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)

class StandardFormatter(logging.Formatter):
    """Standard text formatter"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def get_logger(name: str, level: int = logging.INFO, structured_format: bool = False) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        structured_format: Use JSON structured format if True
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set formatter
    if structured_format:
        formatter = StructuredFormatter()
    else:
        formatter = StandardFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Set custom LogRecord factory
    logging.setLogRecordFactory(CustomLogRecord)
    
    return logger

def get_trading_logger(name: str) -> logging.Logger:
    """Get a logger configured specifically for trading operations"""
    return get_logger(f"trading.{name}", level=logging.INFO)

def get_ml_logger(name: str) -> logging.Logger:
    """Get a logger configured specifically for ML operations"""
    return get_logger(f"ml.{name}", level=logging.INFO, structured_format=True)

def log_trade_event(logger: logging.Logger, event: str, symbol: str, data: Dict[str, Any]):
    """Log a structured trading event"""
    logger.info(
        f"Trade Event: {event} - {symbol}",
        extra={
            'extra_data': {
                'event_type': event,
                'symbol': symbol,
                'trade_data': data
            }
        }
    )

def log_signal_event(logger: logging.Logger, event: str, symbol: str, score: float, data: Dict[str, Any]):
    """Log a structured signal event"""
    logger.info(
        f"Signal Event: {event} - {symbol} (Score: {score})",
        extra={
            'extra_data': {
                'event_type': event,
                'symbol': symbol,
                'signal_score': score,
                'signal_data': data
            }
        }
    )

def log_ml_event(logger: logging.Logger, event: str, model_type: str, data: Dict[str, Any]):
    """Log a structured ML event"""
    logger.info(
        f"ML Event: {event} - {model_type}",
        extra={
            'extra_data': {
                'event_type': event,
                'model_type': model_type,
                'ml_data': data
            }
        }
    )

# Default logger for the application
default_logger = get_logger('algotrader')