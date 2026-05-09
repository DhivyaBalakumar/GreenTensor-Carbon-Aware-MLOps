@echo off
python test_opt_simple.py
if %ERRORLEVEL% NEQ 0 (
    echo Script failed with error code %ERRORLEVEL%
    pause
)
