#!/bin/bash

# 🧪 HISYNC AI - Deployment Test Script
# © 2025 Hire Synchronisation Pvt. Ltd.
# Developed by: Abhishek Rajput (@abhi-hisync)

echo "🧪 HISYNC AI - Deployment Test"
echo "=============================="
echo ""

# Test 1: Local API (should work)
echo "🔍 Test 1: Local API Health Check"
echo "Testing: http://localhost:8000/health"
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo "✅ Local API: WORKING"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "❌ Local API: FAILED"
    echo "   Make sure API is running: python main.py"
fi
echo ""

# Test 2: Domain access (nginx issue)
echo "🔍 Test 2: Domain Health Check"
echo "Testing: https://ai.hisync.in/health"
if curl -s -f https://ai.hisync.in/health > /dev/null 2>&1; then
    DOMAIN_RESPONSE=$(curl -s https://ai.hisync.in/health)
    echo "✅ Domain API: WORKING"
    echo "   Response: $DOMAIN_RESPONSE"
else
    echo "❌ Domain API: FAILED"
    echo "   Issue: Nginx configuration problem"
    echo "   Solution: Update nginx config in Laravel Forge"
fi
echo ""

# Test 3: Check latest changes
echo "🔍 Test 3: API Version Check"
echo "Testing: http://localhost:8000/company"
if curl -s -f http://localhost:8000/company > /dev/null 2>&1; then
    COMPANY_RESPONSE=$(curl -s http://localhost:8000/company | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/company)
    echo "✅ Company Info: WORKING"
    echo "   Developer info should be visible in response"
    echo ""
    echo "📋 Company Response:"
    echo "$COMPANY_RESPONSE"
else
    echo "❌ Company endpoint: FAILED"
fi
echo ""

# Test 4: Check if running process
echo "🔍 Test 4: Process Check"
if pgrep -f "python.*main.py" > /dev/null; then
    PID=$(pgrep -f "python.*main.py")
    echo "✅ HISYNC AI Process: RUNNING (PID: $PID)"
else
    echo "❌ HISYNC AI Process: NOT RUNNING"
    echo "   Start with: python main.py"
fi
echo ""

# Summary
echo "📊 Test Summary:"
echo "==============="
echo "Local API:     $(if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then echo "✅ WORKING"; else echo "❌ FAILED"; fi)"
echo "Domain Access: $(if curl -s -f https://ai.hisync.in/health > /dev/null 2>&1; then echo "✅ WORKING"; else echo "❌ FAILED (Nginx Issue)"; fi)"
echo "Process:       $(if pgrep -f "python.*main.py" > /dev/null; then echo "✅ RUNNING"; else echo "❌ STOPPED"; fi)"
echo ""

if ! curl -s -f https://ai.hisync.in/health > /dev/null 2>&1; then
    echo "🔧 SOLUTION FOR DOMAIN ISSUE:"
    echo "============================="
    echo "1. Go to Laravel Forge Dashboard"
    echo "2. Sites → ai.hisync.in → Nginx Configuration"
    echo "3. Replace current config with content from: nginx-forge-config.txt"
    echo "4. Save and reload nginx"
    echo ""
    echo "📋 Key nginx changes needed:"
    echo "• Change: root /home/forge/ai.hisync.in/public"
    echo "• To:     root /home/forge/ai.hisync.in"
    echo "• Remove: PHP handling (fastcgi_pass)"
    echo "• Add:    proxy_pass http://127.0.0.1:8000"
    echo ""
fi

echo "🔥 Test completed!"
echo "© 2025 Hire Synchronisation Pvt. Ltd."
echo "Developed by: Abhishek Rajput (@abhi-hisync)" 