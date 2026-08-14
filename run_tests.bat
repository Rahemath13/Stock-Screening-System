@echo off
title Stock Screening System - Unit Tests
echo ============================================================
echo  Running Automated Unit Tests...
echo ============================================================
".venv\Scripts\python.exe" tests/test_screener_indicators.py
echo.
pause
