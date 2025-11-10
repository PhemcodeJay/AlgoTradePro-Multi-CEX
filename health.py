import os
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
import psycopg2

logger = logging.getLogger(__name__)

app = FastAPI()

class HealthCheck:
    """Production health check monitoring"""

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.last_check = None
        self.status = "starting"

    def check_system(self, trading_engine=None, db_manager=None) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                "checks": {}
            }

            # Database check
            if db_manager:
                try:
                    with db_manager.get_session() as session:
                        session.execute("SELECT 1")
                    health_status["checks"]["database"] = "healthy"
                except Exception as e:
                    health_status["checks"]["database"] = f"unhealthy: {str(e)}"
                    health_status["status"] = "degraded"

            # Trading engine check
            if trading_engine:
                try:
                    is_connected = trading_engine.client.is_connected() if hasattr(trading_engine, 'client') else False
                    health_status["checks"]["exchange_connection"] = "healthy" if is_connected else "disconnected"
                    if not is_connected:
                        health_status["status"] = "degraded"
                except Exception as e:
                    health_status["checks"]["exchange_connection"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"

            self.last_check = datetime.now(timezone.utc)
            self.status = health_status["status"]

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }

# Global health check instance
health_checker = HealthCheck()


@app.get("/health")
async def health_check_endpoint():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "database": False,
            "app": True
        }
    }

    # Check database connection
    try:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            conn = psycopg2.connect(db_url)
            conn.close()
            health_status["checks"]["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["status"] = "degraded"

    return JSONResponse(health_status)

@app.get("/")
async def root():
    return {"message": "AlgoTraderPro API", "version": "2.0"}