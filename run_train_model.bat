@echo off
title Stock Screening System - Retrain ML Model
echo ============================================================
echo  Retraining AI/ML Model in Python 3.11 Environment...
echo ============================================================
".venv\Scripts\python.exe" scripts/train_model.py
echo.
pause
