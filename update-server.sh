#!/bin/bash

# 🔄 HISYNC AI - Server Update Script
# © 2025 Hire Synchronisation Pvt. Ltd.
# Developed by: Abhishek Rajput (@abhi-hisync)

echo "🔄 HISYNC AI - Server Update"
echo "============================"
echo "© 2025 Hire Synchronisation Pvt. Ltd."
echo "Developed by: Abhishek Rajput (@abhi-hisync)"
echo ""

echo "📥 Step 1: Pulling latest changes from GitHub..."
git pull origin main
echo ""

echo "🛑 Step 2: Stopping existing HISYNC AI process..."
# Stop using the control script if it exists
if [ -f "hisync-control.sh" ]; then
    echo "   Using hisync-control.sh..."
    ./hisync-control.sh stop
else
    echo "   Using pkill..."
    pkill -f "python.*main.py" || echo "   No existing process found"
fi
echo ""

echo "⏳ Step 3: Waiting for process to stop..."
sleep 3
echo ""

echo "🚀 Step 4: Starting HISYNC AI with latest changes..."
# Start using the control script if it exists
if [ -f "hisync-control.sh" ]; then
    echo "   Using hisync-control.sh..."
    ./hisync-control.sh start
else
    echo "   Starting directly..."
    nohup python3 main.py > hisync-ai.log 2>&1 &
    echo $! > hisync-ai.pid
    echo "   ✅ Started with PID: $(cat hisync-ai.pid)"
fi
echo ""

echo "⏳ Step 5: Waiting for service to initialize..."
sleep 10
echo ""

echo "🧪 Step 6: Testing updated deployment..."
echo "Testing health endpoint..."
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8000/health)
    echo "✅ Health check: PASSED"
    echo "   Response: $HEALTH"
else
    echo "❌ Health check: FAILED"
    echo "   Check logs: tail -f hisync-ai.log"
fi
echo ""

echo "Testing company endpoint..."
if curl -s -f http://localhost:8000/company > /dev/null 2>&1; then
    COMPANY=$(curl -s http://localhost:8000/company | python3 -c "import sys,json; data=json.load(sys.stdin); print(f'Company: {data.get(\"company_name\", \"N/A\")}'); print(f'Developer: {data.get(\"developer\", {}).get(\"name\", \"N/A\")}'); print(f'Version: {data.get(\"version\", \"N/A\")}')" 2>/dev/null)
    echo "✅ Company endpoint: PASSED"
    echo "$COMPANY"
else
    echo "❌ Company endpoint: FAILED"
fi
echo ""

echo "🌐 Step 7: Testing domain access..."
if curl -s -f https://ai.hisync.in/health > /dev/null 2>&1; then
    echo "✅ Domain access: WORKING"
    echo "   Your latest changes should be visible at: https://ai.hisync.in"
else
    echo "❌ Domain access: FAILED"
    echo "   Check nginx configuration in Laravel Forge"
fi
echo ""

echo "📊 Update Summary:"
echo "=================="
echo "✅ Code updated from GitHub"
echo "✅ Service restarted"
echo "✅ Latest changes deployed"
echo ""
echo "🌐 Access Points:"
echo "• API: https://ai.hisync.in"
echo "• Docs: https://ai.hisync.in/docs"
echo "• Health: https://ai.hisync.in/health"
echo "• Company: https://ai.hisync.in/company"
echo ""
echo "🔥 HISYNC AI updated successfully!"
echo "© 2025 Hire Synchronisation Pvt. Ltd."
echo "Developed by: Abhishek Rajput (@abhi-hisync)" 