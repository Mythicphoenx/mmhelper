@echo off
rem
rem Simple run script for Windows Systems
rem
rem This script assumes that Python has been correctly setup for your shell. 
rem


rem WHERE python
python -V >nul 2>&1

rem if %ERRORLEVEL% NEQ 0(
rem     ECHO The Python command, python, is not recognized by your terminal. 
rem     ECHO Please setup Python properly then run this command again.
rem     &ECHO.
rem     ECHO You may now close this terminal.
rem     pause
rem     exit /b %errorlevel%
rem )

python -m mmhelper.gui
