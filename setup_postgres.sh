  GNU nano 7.2                    setup_postgres.sh
#!/bin/bash
set -e

# === PostgreSQL Database Configuration ===
DB_HOST="algotrader-db.ctuauq84wu6f.eu-north-1.rds.amazonaws.com"   # 👈 replac>
DB_PORT=5432
DB_NAME="algotrader-db"
DB_USER="postgres"
DB_PASSWORD="algotrader1234"

# === Export environment variables for your bot/app ===
echo "Exporting PostgreSQL environment variables..."
cat <<EOF | sudo tee /etc/profile.d/postgres_env.sh > /dev/null
export DB_HOST="$DB_HOST"
export DB_PORT="$DB_PORT"
export DB_NAME="$DB_NAME"
export DB_USER="$DB_USER"
export DB_PASSWORD="$DB_PASSWORD"
export DATABASE_URL="postgresql+psycopg2://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_P>
EOF
