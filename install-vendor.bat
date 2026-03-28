@echo off
REM Install dependencies from vendored packages (offline install)
REM Usage: install-vendor.bat

setlocal enabledelayedexpansion

set VENDOR_DIR=%~dp0vendor

if not exist "%VENDOR_DIR%" (
    echo ERROR: vendor directory not found at %VENDOR_DIR%
    exit /b 1
)

echo Installing from vendor directory: %VENDOR_DIR%
for /f %%A in ('dir /b "%VENDOR_DIR%\*.whl" "%VENDOR_DIR%\*.tar.gz" 2^>nul ^| find /c /v ""') do set PKG_COUNT=%%A
echo Found %PKG_COUNT% packages

python -m pip install ^
    --no-index ^
    --find-links "%VENDOR_DIR%" ^
    "%VENDOR_DIR%\*.whl"

echo.
echo ✓ Installation complete
echo.
