# run-weekend.ps1
# Lanza ml.automation.weekend y deja un log diario. Pensado para el Task Scheduler
# de Windows, cada hora de viernes a lunes (ver registro al final de este fichero).

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

# Misma precaucion que run-weather-ingestion.ps1: el jar Java lee MONGO_DB del
# entorno real, no de .env. Python si lee .env (MONGODB_DB).
$env:MONGO_DB = "F1-WeatherRec-Prod"

$logDir = Join-Path $projectRoot "ml\automation\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir ("weekend-" + (Get-Date -Format "yyyy-MM-dd") + ".log")

$python = Join-Path $projectRoot "venv\Scripts\python.exe"
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] inicio" | Add-Content -Path $logFile
Set-Location $projectRoot
cmd /c "`"$python`" -m ml.automation.weekend >> `"$logFile`" 2>&1"
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] fin (codigo $LASTEXITCODE)" | Add-Content -Path $logFile

# Registro de la tarea (una vez, desde PowerShell en la raiz del repo):
#   $action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PWD\ml\automation\run-weekend.ps1`""
#   $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday,Saturday,Sunday,Monday -At 06:00
#   $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At 06:00 -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Hours 18)).Repetition
#   Register-ScheduledTask -TaskName "F1WeatherRec-Weekend" -Action $action -Trigger $trigger -Description "Predicciones y dashboard del fin de semana de carrera"
