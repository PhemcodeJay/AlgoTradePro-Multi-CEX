
import os
from datetime import datetime, timezone

def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exchange": os.getenv("EXCHANGE", "binance"),
        "mode": os.getenv("TRADING_MODE", "virtual")
    }
