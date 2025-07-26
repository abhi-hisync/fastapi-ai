#!/bin/bash

# HISYNC AI - No-Sudo Deployment Script for Laravel Forge
# © 2024 HISYNC Technologies. All rights reserved.
# 
# This script deploys HISYNC AI WITHOUT requiring sudo access
# Perfect for shared hosting or limited access servers

set -e  # Exit on any error

echo "🔥 HISYNC AI - No-Sudo Deployment Script"
echo "========================================="
echo "© 2024 HISYNC Technologies"
echo "Deploying to: ai.hisync.in (No sudo required!)"
echo ""

# Configuration
DOMAIN="ai.hisync.in"
APP_DIR="/home/forge/$DOMAIN"
SERVICE_NAME="hisync-ai"

# Auto-detect Python version
PYTHON_CMD=""
for py_version in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v $py_version >/dev/null 2>&1; then
        PYTHON_CMD=$py_version
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ No Python 3 found! Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)

echo "📋 Deployment Configuration (No-Sudo Mode):"
echo "   Domain: $DOMAIN"
echo "   Directory: $APP_DIR"
echo "   Python: $PYTHON_CMD ($PYTHON_VERSION)"
echo "   User: $(whoami)"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found! Please run this script from the HISYNC AI project directory."
    echo "   Current directory: $(pwd)"
    echo "   Expected files: main.py, requirements.txt, image_classifier.py"
    exit 1
fi

# Check if virtual environment module is available
echo "🐍 Checking Python virtual environment support..."
VENV_CMD="venv"

if $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
    echo "✅ Python venv module available"
    VENV_CMD="venv"
elif $PYTHON_CMD -m virtualenv --help >/dev/null 2>&1; then
    echo "✅ Python virtualenv module available"
    VENV_CMD="virtualenv"
else
    echo "⚠️  Virtual environment not available, installing locally..."
    echo "📦 Installing virtualenv package..."
    
    # Try to install virtualenv
    if $PYTHON_CMD -m pip install --user virtualenv --quiet; then
        echo "✅ virtualenv installed successfully"
        VENV_CMD="virtualenv"
    else
        echo "❌ Failed to install virtualenv"
        echo "🔍 Checking system packages..."
        
        # Check if python3-venv is available (Ubuntu/Debian)
        if command -v apt-get >/dev/null 2>&1; then
            echo "💡 Try: apt-get install python3-venv (ask your hosting provider)"
        fi
        
        echo "❌ Cannot proceed without virtual environment support"
        exit 1
    fi
fi

echo "🔧 Using virtual environment method: $VENV_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "🐍 Creating Python virtual environment..."
    # Remove any incomplete venv directory
    rm -rf venv 2>/dev/null || true
    
    if [ "$VENV_CMD" = "virtualenv" ]; then
        $PYTHON_CMD -m virtualenv venv
    else
        $PYTHON_CMD -m venv venv
    fi
    
    # Verify virtual environment was created properly
    if [ ! -f "venv/bin/activate" ]; then
        echo "❌ Failed to create virtual environment with $PYTHON_CMD"
        echo "🔄 Trying alternative method..."
        
        # Try with --user virtualenv
        $PYTHON_CMD -m pip install --user virtualenv --quiet
        $PYTHON_CMD -m virtualenv venv
        
        if [ ! -f "venv/bin/activate" ]; then
            echo "❌ Virtual environment creation failed!"
            echo "📋 Available Python versions:"
            ls /usr/bin/python* 2>/dev/null || echo "No Python found in /usr/bin/"
            exit 1
        fi
    fi
    echo "✅ Virtual environment created successfully"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo "📦 Installing HISYNC AI dependencies..."
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Virtual environment activation file not found!"
    echo "🔍 Checking venv directory structure..."
    ls -la venv/ 2>/dev/null || echo "venv directory not found"
    exit 1
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "   Installing from requirements.txt..."
    pip install -r requirements.txt --quiet
    echo "✅ Dependencies installed successfully"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

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
SECRET_KEY=$(openssl rand -base64 32 2>/dev/null || echo "hisync-$(date +%s)-secret-key")

# Performance Settings
MAX_WORKERS=2
TIMEOUT=60

# Support Information
SUPPORT_EMAIL=support@hisync.in
COMPANY_WEBSITE=https://hisync.in
EOF
    echo "✅ Environment file created with secure settings"
else
    echo "✅ Environment file already exists"
fi

# Create a simple process management script (no systemd/supervisor needed)
echo "🔧 Creating HISYNC AI process manager..."
cat > start-hisync.sh << 'EOF'
#!/bin/bash
# HISYNC AI Process Manager (No sudo required)

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$APP_DIR/hisync-ai.pid"
LOG_FILE="$APP_DIR/hisync-ai.log"

start_hisync() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "🔥 HISYNC AI is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    echo "🚀 Starting HISYNC AI..."
    cd "$APP_DIR"
    source venv/bin/activate
    nohup python main.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "✅ HISYNC AI started (PID: $(cat "$PID_FILE"))"
    echo "📝 Logs: tail -f $LOG_FILE"
}

stop_hisync() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "🛑 Stopping HISYNC AI..."
        kill "$(cat "$PID_FILE")"
        rm -f "$PID_FILE"
        echo "✅ HISYNC AI stopped"
    else
        echo "⚠️  HISYNC AI is not running"
    fi
}

status_hisync() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "✅ HISYNC AI is running (PID: $(cat "$PID_FILE"))"
        echo "📊 Memory usage: $(ps -o pid,rss,vsz,comm -p "$(cat "$PID_FILE")" | tail -n 1)"
    else
        echo "❌ HISYNC AI is not running"
    fi
}

restart_hisync() {
    stop_hisync
    sleep 2
    start_hisync
}

case "$1" in
    start)
        start_hisync
        ;;
    stop)
        stop_hisync
        ;;
    restart)
        restart_hisync
        ;;
    status)
        status_hisync
        ;;
    logs)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "🔥 HISYNC AI Process Manager"
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start HISYNC AI service"
        echo "  stop    - Stop HISYNC AI service"
        echo "  restart - Restart HISYNC AI service"
        echo "  status  - Check HISYNC AI status"
        echo "  logs    - View HISYNC AI logs"
        exit 1
        ;;
esac
EOF

chmod +x start-hisync.sh

# Create a simple auto-restart script
echo "🔄 Creating auto-restart script..."
cat > auto-restart.sh << 'EOF'
#!/bin/bash
# HISYNC AI Auto-Restart Script (Add to crontab)

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$APP_DIR/hisync-ai.pid"

# Check if HISYNC AI is running
if [ ! -f "$PID_FILE" ] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "$(date): HISYNC AI not running, restarting..." >> "$APP_DIR/auto-restart.log"
    cd "$APP_DIR"
    ./start-hisync.sh start
fi
EOF

chmod +x auto-restart.sh

# Stop any existing process
echo "🛑 Stopping any existing HISYNC AI process..."
./start-hisync.sh stop 2>/dev/null || true

# Start HISYNC AI
echo "▶️  Starting HISYNC AI service..."
./start-hisync.sh start

# Wait for service to start
echo "⏳ Waiting for HISYNC AI to initialize..."
sleep 10

# Test the deployment
echo "🧪 Testing HISYNC AI deployment..."

# Test local connection
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Local health check passed"
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "❌ Local health check failed"
    echo "📝 Checking logs..."
    tail -n 20 hisync-ai.log
    exit 1
fi

# Test company endpoint
if curl -s -f http://localhost:8000/company > /dev/null 2>&1; then
    echo "✅ Company endpoint working"
else
    echo "⚠️  Company endpoint check failed"
fi

# Display service status
echo ""
echo "📊 HISYNC AI Service Status:"
./start-hisync.sh status

# Create nginx configuration suggestion
echo ""
echo "📋 Nginx Configuration for Laravel Forge:"
echo "   (Add this to your site's nginx config in Forge dashboard)"
echo ""
cat > nginx-config.txt << 'EOF'
# Add this to your nginx server block in Laravel Forge

location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # Handle large file uploads for HISYNC AI
    client_max_body_size 10M;
    
    # Timeouts for AI processing
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
}

# Health check endpoint
location /health {
    proxy_pass http://127.0.0.1:8000/health;
    access_log off;
}
EOF

echo "📄 Nginx config saved to: nginx-config.txt"

# Setup auto-restart cron job suggestion
echo ""
echo "🕐 Auto-Restart Setup (Optional):"
echo "   Add this to your crontab to auto-restart if needed:"
echo "   */5 * * * * cd $(pwd) && ./auto-restart.sh"
echo ""
echo "   To add to crontab: crontab -e"
echo "   Then add the line above"

# Final instructions
echo ""
echo "🎉 HISYNC AI No-Sudo Deployment Complete!"
echo "========================================"
echo ""
echo "🌐 Access Points:"
echo "   • Local API: http://localhost:8000"
echo "   • Health Check: http://localhost:8000/health"
echo "   • Documentation: http://localhost:8000/docs"
echo "   • Company Info: http://localhost:8000/company"
echo ""
echo "📊 Management Commands:"
echo "   • Check Status: ./start-hisync.sh status"
echo "   • View Logs: ./start-hisync.sh logs"
echo "   • Restart: ./start-hisync.sh restart"
echo "   • Stop: ./start-hisync.sh stop"
echo ""
echo "📝 Important Files Created:"
echo "   • start-hisync.sh - Process manager"
echo "   • auto-restart.sh - Auto-restart script"
echo "   • nginx-config.txt - Nginx configuration"
echo "   • hisync-ai.log - Application logs"
echo "   • .env - Environment configuration"
echo ""
echo "🔧 Next Steps:"
echo "   1. Copy nginx-config.txt content to your Forge site's nginx config"
echo "   2. Set up SSL certificate in Laravel Forge dashboard"
echo "   3. Point your domain ai.hisync.in to this server"
echo "   4. Optional: Add auto-restart.sh to crontab"
echo ""
echo "📞 Support:"
echo "   • Email: support@hisync.in"
echo "   • Website: https://hisync.in"
echo ""
echo "🔥 HISYNC AI is now running without sudo!"
echo "© 2024 HISYNC Technologies. All rights reserved." 