#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════
# SMC Multi-Asset Trading Bot — Server Deployment Script
# ═══════════════════════════════════════════════════════
# Usage: scp deploy.sh root@server:~ && ssh root@server 'bash deploy.sh'
# After: scp models/*.pkl root@server:~/bot/models/
#        scp models/*.zip root@server:~/bot/models/

BOT_DIR="/root/bot"
REPO_URL="https://github.com/laurinvalerian/smc_crypto.git"
PYTHON_VERSION="3.11"

echo "=== SMC Bot Deployment ==="

# 1. System packages
apt-get update -qq
apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip git ufw

# 2. Clone or pull repo
if [ -d "$BOT_DIR" ]; then
    cd "$BOT_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$BOT_DIR"
fi
cd "$BOT_DIR"

# 3. Python venv + deps
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# ML deps (lightweight — no torch/SB3 on server, only inference)
pip install xgboost pyarrow

# 4. Create directories
mkdir -p models trade_journal live_results backtest/results/rl

# 5. Environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "WARNING: Edit .env with your API keys before starting!"
fi

# 6. Firewall: allow SSH + dashboard
ufw allow 22/tcp
ufw allow 8080/tcp
ufw --force enable

# 7. Systemd service — trading bot
cat > /etc/systemd/system/trading-bot.service << 'UNIT'
[Unit]
Description=SMC Multi-Asset Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/bot
ExecStart=/root/bot/venv/bin/python3 live_multi_bot.py
Restart=always
RestartSec=10
StandardOutput=append:/root/bot/paper_trading.log
StandardError=append:/root/bot/paper_trading.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

# 8. Systemd service — dashboard
cat > /etc/systemd/system/trading-dashboard.service << 'UNIT'
[Unit]
Description=SMC Trading Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/bot
ExecStart=/root/bot/venv/bin/python3 dashboard.py
Restart=always
RestartSec=5
StandardOutput=append:/root/bot/dashboard.log
StandardError=append:/root/bot/dashboard.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT

# 9. Enable services (don't start until models are deployed)
systemctl daemon-reload
systemctl enable trading-bot trading-dashboard

# 10. Swap file (4GB for safety)
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
fi

echo ""
echo "=== Deployment Complete ==="
echo "Next steps:"
echo "  1. Edit .env with API keys"
echo "  2. Copy models: scp models/*.pkl root@SERVER:~/bot/models/"
echo "  3. Copy DQN:    scp models/*.zip root@SERVER:~/bot/models/"
echo "  4. Copy clusters: scp config/instrument_clusters.json root@SERVER:~/bot/config/"
echo "  5. Start bot:     systemctl start trading-bot"
echo "  6. Start dashboard: systemctl start trading-dashboard"
echo "  7. Check logs:    journalctl -u trading-bot -f"
echo "  8. Dashboard:     http://SERVER_IP:8080"
