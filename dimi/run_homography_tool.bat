@echo off
echo ===================================================
echo Homografie Calibratie Tool - Dimi (CV-Challenge)
echo ===================================================
cd /d "%~dp0"
..\venv\Scripts\python.exe calibrate_homography.py
pause
