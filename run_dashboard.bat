@echo off
title Stock Screening System - Streamlit Dashboard
echo ============================================================
echo  Launching AI/ML Stock Screening System Dashboard...
echo ============================================================
".venv\Scripts\python.exe" -m streamlit run src/dashboard/app.py
pause
