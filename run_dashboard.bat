@echo off
echo ======================================================================
echo BSE SENSEX vs Geometric Brownian Motion (GBM) Quantitative Dashboard
echo ======================================================================
echo.
echo Select launch mode:
echo [1] Streamlit Interactive Dashboard (Python App)
echo [2] Standalone Modern HTML Web Visualizer (Default Browser)
echo [3] Regenerate Publication 300-DPI Graphics & Animated GIF
echo.
set /p mode="Enter choice (1, 2, or 3): "

if "%mode%"=="1" (
    echo Starting Streamlit App...
    streamlit run app.py
) else if "%mode%"=="2" (
    echo Opening Standalone HTML Visualizer...
    start index.html
) else if "%mode%"=="3" (
    echo Generating High-Res Charts and Animated GIF...
    python generate_publication_charts.py
    python generate_animation.py
    echo Done!
    pause
) else (
    echo Invalid choice. Opening standalone visualizer by default...
    start index.html
)
