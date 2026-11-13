@echo off
echo ===================================================
echo Starten van Streamlit Dashboard - Dimi
echo ===================================================
cd /d "%~dp0"
..\venv\Scripts\python.exe -m streamlit run app.py
pause
