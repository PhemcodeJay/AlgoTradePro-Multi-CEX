import os
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

from logging_config import get_trading_logger

logger = get_trading_logger('exceptions')

TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

if TYPE_CHECKING:
    from db import DatabaseManager, ErrorLog

class TradingException(Exception):
    """Base exception for trading-related errors"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_exception) if self.original_exception else None,
            'trading_mode': TRADING_MODE
        }

class APIException(TradingException):
    """Base exception for API-related errors"""
    pass

class APIConnectionException(APIException):
    """Exception for API connection issues"""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, context=context, original_exception=original_exception)
        self.endpoint = endpoint
        self.status_code = status_code
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            'endpoint': self.endpoint,
            'status_code': self.status_code
        })
        return data

class APIRateLimitException(APIException):
    """Exception for API rate limit violations"""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, context=context, original_exception=original_exception)
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['retry_after'] = self.retry_after
        return data

class APIAuthenticationException(APIException):
    """Exception for API authentication failures"""
    pass

class APITimeoutException(APIException):
    """Exception for API timeout errors"""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, context=context, original_exception=original_exception)
        self.timeout_duration = timeout_duration
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['timeout_duration'] = self.timeout_duration
        return data

class APIDataException(APIException):
    """Exception for API data validation or parsing errors"""
    
    def __init__(
        self,
        message: str,
        response_data: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, context=context, original_exception=original_exception)
        self.response_data = response_data
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['response_data'] = self.response_data
        return data

class InsufficientFundsError(TradingException):
    """Exception for insufficient account balance"""
    pass

class InvalidOrderError(TradingException):
    """Exception for invalid order parameters"""
    pass

class RiskManagementError(TradingException):
    """Exception for risk management violations"""
    pass

class SignalGenerationError(TradingException):
    """Exception for signal generation failures"""
    pass

class MLFilterError(TradingException):
    """Exception for ML filtering failures"""
    pass

class DatabaseError(TradingException):
    """Exception for database operations"""
    pass

@dataclass
class APIErrorRecoveryStrategy:
    """Strategy for handling API error retries"""
    max_retries: int = 3
    delay: float = 1.0
    backoff_factor: float = 2.0

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if a retry should be attempted"""
        if attempt >= self.max_retries:
            return False
        if isinstance(exception, (APIRateLimitException, APIAuthenticationException)):
            return False
        if isinstance(exception, APIConnectionException) and exception.status_code in (401, 403):
            return False
        return True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        return self.delay * (self.backoff_factor ** attempt)

def create_error_context(
    module: str,
    function: str,
    exchange: Optional[str] = None,
    symbol: Optional[str] = None,
    operation_type: Optional[str] = None,
    trading_mode: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error context"""
    context = {
        'module': module,
        'function': function,
        'timestamp': datetime.utcnow().isoformat(),
        'trading_mode': trading_mode or TRADING_MODE
    }
    if exchange:
        context['exchange'] = exchange
    if symbol:
        context['symbol'] = symbol
    if operation_type:
        context['operation_type'] = operation_type
    if extra_data:
        context['extra_data'] = json.dumps(extra_data)  # Serialize to string
    return context

def handle_trading_exception(
    logger,
    exception: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None
) -> TradingException:
    """Convert generic exceptions to trading exceptions with proper context"""
    if isinstance(exception, TradingException):
        return exception
    
    context = context or {}
    error_str = str(exception).lower()
    
    # Map specific API errors
    if "rate limit" in error_str or "429" in error_str:
        trading_exception = APIRateLimitException(
            f"Rate limit exceeded during {operation}: {str(exception)}",
            retry_after=context.get('extra_data', {}).get('retry_after', 60),
            context=context,
            original_exception=exception
        )
    elif "connection" in error_str or "timeout" in error_str:
        trading_exception = APIConnectionException(
            f"Connection error during {operation}: {str(exception)}",
            endpoint=context.get('extra_data', {}).get('endpoint'),
            status_code=context.get('extra_data', {}).get('status_code'),
            context=context,
            original_exception=exception
        )
    elif "authentication" in error_str or "401" in error_str or "403" in error_str:
        trading_exception = APIAuthenticationException(
            f"Authentication error during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    elif "json" in error_str or "invalid response" in error_str:
        trading_exception = APIDataException(
            f"Data parsing error during {operation}: {str(exception)}",
            response_data=context.get('extra_data', {}).get('response_data'),
            context=context,
            original_exception=exception
        )
    elif "insufficient" in error_str:
        trading_exception = InsufficientFundsError(
            f"Insufficient funds during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    elif "order" in error_str or "invalid parameter" in error_str:
        trading_exception = InvalidOrderError(
            f"Invalid order during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    elif "risk" in error_str or "margin" in error_str:
        trading_exception = RiskManagementError(
            f"Risk management violation during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    elif "signal" in error_str:
        trading_exception = SignalGenerationError(
            f"Signal generation error during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    elif "ml" in error_str or "machine learning" in error_str:
        trading_exception = MLFilterError(
            f"ML filter error during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    elif "database" in error_str or "sql" in error_str:
        trading_exception = DatabaseError(
            f"Database error during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    else:
        trading_exception = APIException(
            f"Unexpected API error during {operation}: {str(exception)}",
            context=context,
            original_exception=exception
        )
    
    logger.error(
        f"Trading exception: {trading_exception.message}",
        extra={'context': trading_exception.to_dict()}
    )
    
    # Log to database if ErrorLog and add_error_log are available
    try:
        from db import DatabaseManager, ErrorLog
        db_manager = DatabaseManager()
        if hasattr(DatabaseManager, 'add_error_log'):
            error_log = ErrorLog(
                error_type=type(trading_exception).__name__,
                message=trading_exception.message,
                error_code=trading_exception.error_code,
                context=trading_exception.context,
                timestamp=trading_exception.timestamp,
                trading_mode=TRADING_MODE
            )
            db_manager.add_error_log(error_log)
            db_manager.session.commit()
        else:
            logger.warning("DatabaseManager.add_error_log not available, skipping database logging")
    except ImportError:
        logger.warning("ErrorLog or DatabaseManager not available, skipping database logging")
    except Exception as db_error:
        logger.error(f"Failed to log error to database: {db_error}")
    finally:
        if 'db_manager' in locals():
            db_manager.session.close()
    
    return trading_exception