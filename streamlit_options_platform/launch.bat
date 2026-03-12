@echo off
echo =====================================
echo Professional Options Trading Platform
echo =====================================
echo.
echo Starting Streamlit application...
echo Platform will be available at: http://localhost:8501
echo.
echo Features:
echo - Real-time portfolio monitoring
echo - Interactive volatility surfaces
echo - Advanced options chain analysis
echo - Risk management dashboards
echo - Options flow and sentiment analysis
echo.
echo Press Ctrl+C to stop the application
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking requirements...
python -c "import streamlit, plotly, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Launch the application
echo Starting application...
streamlit run main.py --server.port=8501 --browser.gatherUsageStats=false

pause