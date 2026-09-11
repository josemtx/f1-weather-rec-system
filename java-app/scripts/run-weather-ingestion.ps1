# run-weather-ingestion.ps1
# Ejecuta la ingesta de pronosticos de clima (AppWeather) y registra el resultado en un log diario.
# Pensado para ser invocado por el Task Scheduler de Windows.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$javaAppDir = Split-Path -Parent $scriptDir
$projectRoot = Split-Path -Parent $javaAppDir

# Si la API key no esta en el entorno (por ejemplo, el Task Scheduler no la propago),
# se recupera desde .env como respaldo.
if (-not $env:OPENWEATHER_API_KEY) {
    $envFile = Join-Path $projectRoot ".env"
    if (Test-Path $envFile) {
        $line = Get-Content $envFile | Select-String '^OPENWEATHER_API_KEY='
        if ($line) {
            $env:OPENWEATHER_API_KEY = $line.ToString().Split('=', 2)[1].Trim()
        }
    }
}

$logDir = Join-Path $javaAppDir "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$logFile = Join-Path $logDir ("weather-ingestion-{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

$jar = Join-Path $javaAppDir "target\F1-WeatherRec.jar"

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Iniciando run-weather-ingestion.ps1" | Add-Content -Path $logFile

if (-not (Test-Path $jar)) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR: no se encontro $jar. Ejecuta 'mvn clean package' en java-app antes de programar esta tarea." | Add-Content -Path $logFile
    exit 1
}

if (-not $env:OPENWEATHER_API_KEY) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR: OPENWEATHER_API_KEY no esta disponible (ni en el entorno ni en .env)." | Add-Content -Path $logFile
    exit 1
}

Set-Location $javaAppDir
# Se usa cmd.exe para la redireccion: PowerShell 5.1 envuelve el stderr de los
# ejecutables nativos en NativeCommandError y aborta con $ErrorActionPreference="Stop",
# aunque el proceso termine con codigo 0. java.util.logging escribe por stderr.
cmd /c "java -jar `"$jar`" weather >> `"$logFile`" 2>&1"
$exitCode = $LASTEXITCODE

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Finalizado con codigo de salida $exitCode" | Add-Content -Path $logFile

exit $exitCode
