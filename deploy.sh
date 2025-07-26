#!/bin/bash

# HISYNC AI - Laravel Forge Deployment Script
# © 2024 HISYNC Technologies. All rights reserved.
# 
# This script automates the deployment of HISYNC AI on Laravel Forge
# Domain: ai.hisync.in

set -e  # Exit on any error

echo "🔥 HISYNC AI - Deployment Script"
echo "================================="
echo "© 2024 HISYNC Technologies"
echo "Deploying to: ai.hisync.in"
echo ""

# Configuration
DOMAIN="ai.hisync.in"
APP_DIR="/home/forge/$DOMAIN"
SERVICE_NAME="hisync-ai"
PYTHON_VERSION="python3.9"

echo "📋 Deployment Configuration:"
echo "   Domain: $DOMAIN"
echo "   Directory: $APP_DIR"
echo "   Service: $SERVICE_NAME"
echo "   Python: $PYTHON_VERSION"
echo ""

# Check if running as forge user
if [ "$USER" != "forge" ]; then
    echo "❌ This script must be run as 'forge' user on Laravel Forge server"
    echo "   Please SSH into your server as forge user and run this script"
    exit 1
fi

# Navigate to app directory
echo "📁 Navigating to application directory..."
cd $APP_DIR

# Pull latest code (if git repo exists)
if [ -d ".git" ]; then
    echo "🔄 Pulling latest HISYNC AI code..."
    git pull origin main
else
    echo "⚠️  No git repository found. Please ensure code is properly deployed."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    $PYTHON_VERSION -m venv venv
fi

# Activate virtual environment and install dependencies
echo "📦 Installing HISYNC AI dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment configuration..."
    cat > .env << EOF
# HISYNC AI Configuration
APP_NAME="HISYNC AI - Image Classification API"
COMPANY="HISYNC Technologies"
DEBUG=False
HOST=0.0.0.0
PORT=8000
SECRET_KEY=$(openssl rand -base64 32)

# Performance Settings
MAX_WORKERS=4
TIMEOUT=60

# Support Information
SUPPORT_EMAIL=support@hisync.in
COMPANY_WEBSITE=https://hisync.in
EOF
    echo "✅ Environment file created with secure settings"
else
    echo "✅ Environment file already exists"
fi

# Create systemd service
echo "🔧 Creating HISYNC AI systemd service..."
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << EOF
[Unit]
Description=HISYNC AI - Image Classification API
After=network.target

[Service]
Type=simple
User=forge
Group=forge
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
ExecStart=$APP_DIR/venv/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
echo "🚀 Enabling HISYNC AI service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# Create supervisor configuration
echo "👁️  Setting up process monitoring..."
sudo tee /etc/supervisor/conf.d/$SERVICE_NAME.conf > /dev/null << EOF
[program:$SERVICE_NAME]
command=$APP_DIR/venv/bin/python main.py
directory=$APP_DIR
user=forge
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/$SERVICE_NAME.log
environment=PATH="$APP_DIR/venv/bin"
EOF

# Update supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Start the service
echo "▶️  Starting HISYNC AI service..."
sudo systemctl stop $SERVICE_NAME 2>/dev/null || true
sudo supervisorctl stop $SERVICE_NAME 2>/dev/null || true
sleep 2
sudo supervisorctl start $SERVICE_NAME

# Wait for service to start
echo "⏳ Waiting for HISYNC AI to initialize..."
sleep 15

# Test the deployment
echo "🧪 Testing HISYNC AI deployment..."

# Test local connection
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "✅ Local health check passed"
else
    echo "❌ Local health check failed"
    exit 1
fi

# Test domain connection (if accessible)
if curl -s -f https://$DOMAIN/health > /dev/null; then
    echo "✅ Domain health check passed"
else
    echo "⚠️  Domain health check failed (this is normal if SSL is still setting up)"
fi

# Display service status
echo ""
echo "📊 HISYNC AI Service Status:"
sudo supervisorctl status $SERVICE_NAME

# Display logs
echo ""
echo "📝 Recent HISYNC AI Logs:"
tail -n 10 /var/log/$SERVICE_NAME.log 2>/dev/null || echo "No logs available yet"

# Final instructions
echo ""
echo "🎉 HISYNC AI Deployment Complete!"
echo "=================================="
echo ""
echo "🌐 Access Points:"
echo "   • API Base: https://$DOMAIN"
echo "   • Documentation: https://$DOMAIN/docs"
echo "   • Health Check: https://$DOMAIN/health"
echo "   • Company Info: https://$DOMAIN/company"
echo ""
echo "📊 Management Commands:"
echo "   • Check Status: sudo supervisorctl status $SERVICE_NAME"
echo "   • View Logs: tail -f /var/log/$SERVICE_NAME.log"
echo "   • Restart: sudo supervisorctl restart $SERVICE_NAME"
echo "   • Stop: sudo supervisorctl stop $SERVICE_NAME"
echo ""
echo "📞 Support:"
echo "   • Email: support@hisync.in"
echo "   • Website: https://hisync.in"
echo ""
echo "🔥 HISYNC AI is now live and ready for business!"
echo "© 2024 HISYNC Technologies. All rights reserved." 