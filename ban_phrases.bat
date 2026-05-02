@echo off

:: ==========================================
:: CONFIGURATION (Change these as needed)
:: ==========================================
set "LLAMA_PORT=8080"
set "PROXY_PORT=5001"
set "LLAMA_HOST=127.0.0.1"

:: Internals
set "VENV_DIR=venv"
set "FLAG_FILE=%VENV_DIR%\.packages_installed"
:: ==========================================

:: Change working directory to the location of this batch file
cd /d "%~dp0"

:: 1. Check if the venv exists
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [SETUP] Virtual environment not found. Creating one now...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        pause
        exit /b
    )
    if exist "%FLAG_FILE%" del "%FLAG_FILE%"
)

:: 2. Install dependencies ONLY if the flag file is missing
if not exist "%FLAG_FILE%" (
    echo [SETUP] Checking/Installing dependencies...
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
    "%VENV_DIR%\Scripts\pip.exe" install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b
    )
    echo. > "%FLAG_FILE%"
)

:: 3. Run the Python script using the variables from above

"%VENV_DIR%\Scripts\python.exe" ban_phrases.py --llama-port %LLAMA_PORT% --proxy-port %PROXY_PORT%

echo.
pause