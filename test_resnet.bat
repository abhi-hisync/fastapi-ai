@echo off
echo 🔥 HISYNC AI - Google ResNet v2 Test
echo =====================================

echo.
echo 🏥 Testing Server Health...
curl -X GET "http://localhost:8000/health"

echo.
echo.
echo 🚀 Testing Google ResNet v2 Endpoint...
echo Creating sample image and testing...

REM Create a simple test using PowerShell to generate image
powershell -Command "Add-Type -AssemblyName System.Drawing; $bmp = New-Object System.Drawing.Bitmap(224, 224); $g = [System.Drawing.Graphics]::FromImage($bmp); $g.Clear([System.Drawing.Color]::FromArgb(139, 69, 19)); $bmp.Save('test_coffee.png', [System.Drawing.Imaging.ImageFormat]::Png); $g.Dispose(); $bmp.Dispose()"

REM Test the ResNet endpoint
curl -X POST "http://localhost:8000/classify/resnet" ^
     -F "image=@test_coffee.png" ^
     -F "expected_label=coffee" ^
     -F "confidence_threshold=0.8"

echo.
echo.
echo ✅ Test completed!
echo 📚 Visit http://localhost:8000/docs for interactive testing
echo 🌐 Visit http://localhost:8000 for web interface

REM Cleanup
del test_coffee.png 2>nul

pause
