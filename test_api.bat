@echo off
echo 🔥 HISYNC AI - API Test Commands
echo © 2025 Hire Synchronisation Pvt. Ltd.
echo ====================================

echo.
echo 🏥 Testing Health Endpoint...
curl -X GET "http://localhost:8000/health"

echo.
echo.
echo 📊 Testing Stats Endpoint...
curl -X GET "http://localhost:8000/stats"

echo.
echo.
echo 🏢 Testing Company Info...
curl -X GET "http://localhost:8000/company"

echo.
echo.
echo 📋 Testing Supported Labels...
curl -X GET "http://localhost:8000/labels"

echo.
echo.
echo ✅ All API tests completed!
echo.
echo 📝 For image classification, use:
echo curl -X POST "http://localhost:8000/classify" ^
echo      -F "file=@your_image.jpg" ^
echo      -F "expected_label=coffee"
echo.
echo 🌐 Web Interface: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
pause
