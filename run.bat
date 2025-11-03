@echo off
setlocal ENABLEEXTENSIONS

REM =====================================================================
REM run.bat - Create .venv, install requirements, run app.py, then open URL
REM =====================================================================

set "APP_URL=http://127.0.0.1:7860"
set "READINESS_TIMEOUT_SECONDS=120"

pushd "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  goto :end
)

if not exist ".venv" (
  echo [INFO] Creating virtual environment in .venv ...
  python -m venv ".venv" || (echo [ERROR] Failed to create venv.& goto :end)
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [ERROR] Virtual environment looks broken: "%VENV_PY%" not found.
  goto :end
)

echo [INFO] Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip || (echo [ERROR] pip upgrade failed.& goto :end)

if exist "requirements.txt" (
  echo [INFO] Installing requirements from requirements.txt ...
  "%VENV_PY%" -m pip install -r requirements.txt || (echo [ERROR] Failed to install requirements.& goto :end)
) else (
  echo [INFO] No requirements.txt found; skipping dependency install.
)

REM --- Start the app in a new window so we can keep polling ---
echo [INFO] Launching app.py ...
start "app.py" "%VENV_PY%" "app.py" %*

REM --- Wait for the server to be ready, then open the browser robustly ---
echo [INFO] Waiting for %APP_URL% to become available (timeout %READINESS_TIMEOUT_SECONDS%s)...
powershell -NoProfile -Command ^
  "$u='%APP_URL%';" ^
  "$deadline=(Get-Date).AddSeconds(%READINESS_TIMEOUT_SECONDS%);" ^
  "while((Get-Date) -lt $deadline){" ^
  "  try{ $r=Invoke-WebRequest -Uri $u -UseBasicParsing -Method Head -TimeoutSec 2; if($r.StatusCode -ge 200 -and $r.StatusCode -lt 500){ exit 0 } }catch{}" ^
  "  Start-Sleep -Seconds 1" ^
  "}; exit 1"

if errorlevel 1 (
  echo [WARN] Could not confirm server readiness. Opening browser anyway...
) else (
  echo [INFO] Server is up. Opening browser...
)

REM >>> Use PowerShell to open the default browser (more reliable than `start "" URL`)
powershell -NoProfile -Command "Start-Process '%APP_URL%'"

:end
if not defined CI pause
popd
endlocal
