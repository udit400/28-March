$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $projectRoot "backend"
$pythonPath = Join-Path (Split-Path $projectRoot -Parent) ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
  throw "Python virtual environment not found at $pythonPath"
}

Set-Location $backendPath
& $pythonPath -m uvicorn main:app --host 127.0.0.1 --port 8000