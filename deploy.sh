#!/bin/bash
set -e

# =============================
# CONFIGURATION VARIABLES
# =============================
APP_NAME="app.py"
APP_DIR="/home/ubuntu/AlgoTradePro-Multi-CEX"
VENV_DIR="venv"
PORT=8501

# Load environment variables (for DB and optional APP_DOMAIN, EMAIL_ADDRESS)
if [ -f /etc/profile.d/postgres_env.sh ]; then
    source /etc/profile.d/postgres_env.sh
    echo "Environment variables loaded from /etc/profile.d/postgres_env.sh"
else
    echo "⚠️ /etc/profile.d/postgres_env.sh not found. Skipping DB setup. Ensure environment variables are set if DB is needed."
fi

# =============================
# UPDATE & INSTALL DEPENDENCIES
# =============================
echo "Updating system and installing required packages..."
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip nginx ufw curl git openssl postgresql-client
if [ -n "$APP_DOMAIN" ] && [ "$APP_DOMAIN" != "example.com" ]; then
    sudo apt install -y certbot python3-certbot-nginx
fi

# =============================
# CLONE REPOSITORY (if not already cloned)
# =============================
if [ ! -d "$APP_DIR" ]; then
    echo "Cloning repository..."
    cd /home/ubuntu
    git clone https://github.com/PhemcodeJay/AlgoTradePro-Multi-CEX.git
else
    echo "Repository already exists. Pulling latest changes..."
    cd "$APP_DIR"
    git pull
fi

cd "$APP_DIR"

# =============================
# SETUP PYTHON ENVIRONMENT
# =============================
echo "Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install psycopg2-binary sqlalchemy  # Ensure DB dependencies are installed

# =============================
# DATABASE SETUP (if env vars provided)
# =============================
if [ -n "$DB_HOST" ] && [ -n "$DB_PASSWORD" ] && [ -n "$DB_USER" ] && [ -n "$DB_NAME" ]; then
    echo "Using DATABASE_URL: $DB_HOST:$DB_PORT/$DB_NAME"
    
    # Test RDS connection
    echo "Testing connection to PostgreSQL RDS..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d postgres -p "${DB_PORT:-5432}" -c "\q" 2>/dev/null \
        && echo "✅ RDS connection successful." \
        || { echo "❌ Failed to connect to PostgreSQL RDS. Check endpoint, credentials, or security group."; exit 1; }

    # Create database if missing
    DB_EXIST=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';" 2>/dev/null)
    if [ "$DB_EXIST" = "1" ]; then
        echo "Database '$DB_NAME' already exists."
    else
        echo "Creating database '$DB_NAME'..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d postgres -c "CREATE DATABASE \"$DB_NAME\";" -p "${DB_PORT:-5432}"
        echo "✅ Database '$DB_NAME' created."
    fi
else
    echo "⚠️ DB environment variables not set. Skipping DB setup."
fi

# =============================
# STREAMLIT CONFIG
# =============================
mkdir -p ~/.streamlit
cat <<EOF > ~/.streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT
address = "127.0.0.1"
EOF

# =============================
# SYSTEMD SERVICE FOR STREAMLIT
# =============================
echo "Setting up systemd service for Streamlit..."
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=Streamlit AlgoTradePro App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/$VENV_DIR/bin/streamlit run $APP_DIR/$APP_NAME --server.port=$PORT --server.address=127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl restart streamlit

# =============================
# NGINX + SSL SETUP
# =============================
IP=$(curl -s http://checkip.amazonaws.com)
if [ -n "$APP_DOMAIN" ] && [ "$APP_DOMAIN" != "example.com" ] && [ -n "$EMAIL_ADDRESS" ]; then
    echo "Configuring Nginx for domain $APP_DOMAIN with Let's Encrypt SSL..."
    
    # Nginx config for domain
    sudo tee /etc/nginx/sites-available/streamlit > /dev/null <<EOF
server {
    listen 80;
    server_name $APP_DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

    sudo ln -sf /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl restart nginx

    # Let's Encrypt
    sudo certbot --nginx -d "$APP_DOMAIN" --non-interactive --agree-tos -m "$EMAIL_ADDRESS"
    echo "✅ Nginx + SSL configured for $APP_DOMAIN"
    
    ACCESS_URL="https://$APP_DOMAIN"
else
    echo "Configuring Nginx with self-signed SSL for IP access..."
    
    # Self-signed cert
    sudo mkdir -p /etc/ssl/selfsigned
    sudo openssl req -x509 -nodes -days 730 -newkey rsa:2048 \
      -keyout /etc/ssl/selfsigned/streamlit.key \
      -out /etc/ssl/selfsigned/streamlit.crt \
      -subj "/CN=$IP"

    # Nginx config for IP (self-signed)
    sudo tee /etc/nginx/sites-available/streamlit > /dev/null <<EOF
server {
    listen 443 ssl;
    server_name _;

    ssl_certificate /etc/ssl/selfsigned/streamlit.crt;
    ssl_certificate_key /etc/ssl/selfsigned/streamlit.key;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}

server {
    listen 80;
    return 301 https://\$host\$request_uri;
}
EOF

    sudo ln -sf /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl restart nginx
    
    ACCESS_URL="https://$IP"
    echo "⚠️ Self-signed certificate in use. Browser may show security warning."
fi

# =============================
# FIREWALL CONFIG
# =============================
echo "Configuring UFW firewall..."
sudo ufw allow 22
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

# =============================
# DONE
# =============================
echo ""
echo "✅ Deployment complete!"
echo "🌍 Visit your app: $ACCESS_URL"
echo "📡 Use this IP for Binance whitelist: $IP"
echo "🔍 Check Streamlit logs: sudo journalctl -u streamlit -f"
echo "📋 Nginx logs: sudo tail -f /var/log/nginx/error.log"