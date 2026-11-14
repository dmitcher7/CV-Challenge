@echo off
echo ========================================================
echo  Starting Team 3 - Sports Vision Streamlit Dashboard
echo ========================================================
cd /d "%~dp0final_upload\streamlit"
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -m streamlit run app/app.py
) else (
    python -m streamlit run app/app.py
)
pause
