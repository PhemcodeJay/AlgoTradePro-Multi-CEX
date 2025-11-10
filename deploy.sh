#!/bin/bash
set -e

# ====================================================
# 🚀 AlgoTradePro + DuckDNS + SSL Deployment Script
# Compatible with Amazon Linux 2023
# ====================================================

# -----------------------------
# CONFIGURATION
# -----------------------------
APP_NAME="app.py"
APP_DIR="/home/ec2-user/AlgoTradePro-Multi-CEX"
VENV_DIR="venv"
PORT=5000
REPO_URL="https://github.com/PhemcodeJay/AlgoTradePro-Multi-CEX.git"

# 🔐 DuckDNS Setup
DUCKDNS_DOMAIN="algotraderbot.duckdns.org"
DUCKDNS_TOKEN="946a1430-e8df-4567-ad89-1d7fe6f29ffb"
EMAIL="admin@algotraderbot.duckdns.org"

# -----------------------------
# LOGGING
# -----------------------------
log() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $1\n"; }

# -----------------------------
# SYSTEM PREP
# -----------------------------
log "🔧 Updating and installing dependencies..."
sudo dnf update -y
sudo dnf install -y python3 python3-pip git nginx curl openssl cronie firewalld

sudo pip3 install --upgrade pip
sudo pip3 install certbot certbot-nginx streamlit psycopg2-binary sqlalchemy

# -----------------------------
# DUCKDNS SETUP
# -----------------------------
log "🦆 Configuring DuckDNS..."
sudo mkdir -p /opt/duckdns
sudo tee /opt/duckdns/duck.sh > /dev/null <<EOF
echo url="https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=" | curl -k -o /opt/duckdns/duck.log -K -
EOF
sudo chmod 700 /opt/duckdns/duck.sh
sudo bash /opt/duckdns/duck.sh

sudo systemctl enable crond --now
( sudo crontab -l 2>/dev/null; echo "*/5 * * * * /opt/duckdns/duck.sh >/dev/null 2>&1" ) | sudo crontab -

# -----------------------------
# CLONE OR UPDATE REPO
# -----------------------------
if [ ! -d "$APP_DIR" ]; then
    log "📦 Cloning repo..."
    cd /home/ec2-user
    git clone "$REPO_URL"
else
    log "🔄 Updating existing repo..."
    cd "$APP_DIR"
    git pull
fi

cd "$APP_DIR"

# -----------------------------
# MOVE .env SECURELY
# -----------------------------
if [ -f "/home/ec2-user/.env" ]; then
    log "🔐 Moving .env..."
    mv /home/ec2-user/.env "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
else
    log "⚠️ No .env file found — skipping."
fi

# -----------------------------
# PYTHON ENVIRONMENT
# -----------------------------
log "🐍 Setting up Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
log "🧩 Configuring Streamlit..."
mkdir -p ~/.streamlit
cat <<EOF > ~/.streamlit/config.toml
[server]
headless = true
enableCORS = false
port = $PORT
address = "127.0.0.1"
EOF

# -----------------------------
# SYSTEMD SERVICE
# -----------------------------
log "⚙️ Setting up systemd service..."
sudo tee /etc/systemd/system/algotraderpro.service > /dev/null <<EOF
[Unit]
Description=AlgoTradePro Streamlit App
After=network.target

[Service]
User=ec2-user
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/$VENV_DIR/bin/streamlit run $APP_DIR/$APP_NAME --server.port=$PORT --server.address=127.0.0.1
Restart=always
Environment="PATH=$APP_DIR/$VENV_DIR/bin:/usr/local/bin:/usr/bin"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable algotraderpro
sudo systemctl restart algotraderpro

# -----------------------------
# NGINX CONFIGURATION
# -----------------------------
log "🌐 Configuring Nginx reverse proxy..."
sudo tee /etc/nginx/conf.d/algotraderpro.conf > /dev/null <<EOF
server {
    listen 80;
    server_name ${DUCKDNS_DOMAIN};

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo nginx -t && sudo systemctl restart nginx && sudo systemctl enable nginx

# -----------------------------
# SSL CONFIGURATION
# -----------------------------
log "🔒 Setting up SSL with Certbot..."
sudo certbot --nginx -d ${DUCKDNS_DOMAIN} --non-interactive --agree-tos -m ${EMAIL} --redirect || true
( sudo crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet" ) | sudo crontab -

# -----------------------------
# FIREWALL RULES
# -----------------------------
log "🔥 Configuring firewall..."
sudo systemctl enable firewalld --now
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload

# -----------------------------
# DONE
# -----------------------------
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com)
log "✅ Deployment complete!"
echo "--------------------------------------------"
echo "🌍 Visit your app: https://${DUCKDNS_DOMAIN}"
echo "🔹 IP Address (for Binance whitelist): $PUBLIC_IP"
echo "🔹 Logs: sudo journalctl -u algotraderpro -f"
echo "🔹 DuckDNS Log: /opt/duckdns/duck.log"
echo "--------------------------------------------"
