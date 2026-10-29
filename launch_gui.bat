@echo off
setlocal
cd /d "%~dp0"

echo [*] Checking for Python 3 installation...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [+] Found Python in PATH. Starting LabPBR 1.3 Web GUI...
    python web_ui.py
    goto :eof
)

py -3 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [+] Found Python via 'py' launcher. Starting LabPBR 1.3 Web GUI...
    py -3 web_ui.py
    goto :eof
)

echo [!] Python 3 was not detected on this system.
echo [*] Opening GUI with Python download instructions...
start "" "%~dp0index.html?python=missing"
pause

