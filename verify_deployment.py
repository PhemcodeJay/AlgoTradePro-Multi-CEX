
#!/usr/bin/env python3
"""
Deployment verification script for AlgoTraderPro
Checks all critical components before deployment
"""

import os
import sys
import psycopg2
from sqlalchemy import create_engine, text

def check_env_vars():
    """Verify required environment variables"""
    required = ["DATABASE_URL"]
    missing = [var for var in required if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    print("✅ Environment variables configured")
    return True

def check_database():
    """Test database connectivity"""
    try:
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def check_tables():
    """Verify database tables exist"""
    try:
        from db import db_manager
        db_manager.create_tables()
        print("✅ Database tables verified")
        return True
    except Exception as e:
        print(f"❌ Database tables check failed: {e}")
        return False

def main():
    print("\n🔍 AlgoTraderPro Deployment Verification\n")
    print("=" * 50)
    
    checks = [
        ("Environment Variables", check_env_vars),
        ("Database Connection", check_database),
        ("Database Tables", check_tables),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All checks passed! Ready for deployment.")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
