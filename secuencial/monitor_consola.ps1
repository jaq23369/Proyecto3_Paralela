# Monitor de recursos en consola para la evidencia de ejecucion.
# Muestra, cada segundo y para CADA núcleo lógico, el uso del CPU (contadores de rendimiento de Windows, la misma
# fuente que usa el Administrador de Tareas), la memoria y el estado de la GPU (nvidia-smi).
#
# Uso:  powershell -NoProfile -ExecutionPolicy Bypass -File secuencial\monitor_consola.ps1
param([string]$Titulo = "Monitor de recursos (evidencia)")

$Host.UI.RawUI.WindowTitle = $Titulo
try { $Host.UI.RawUI.BufferSize = New-Object System.Management.Automation.Host.Size(96, 40);
      $Host.UI.RawUI.WindowSize = New-Object System.Management.Automation.Host.Size(96, 40) } catch {}
$cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name.Trim()
$ramTotal = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Clear-Host
$bloque = [string][char]0x2588; $vacio = [string][char]0x2591

while ($true) {
    $s = Get-Counter -Counter '\Processor(*)\% Processor Time', '\Memory\Available MBytes' -ErrorAction SilentlyContinue
    $nucleos = $s.CounterSamples | Where-Object { $_.Path -like '*processor(*' -and $_.InstanceName -ne '_total' } |
        Sort-Object { [int]$_.InstanceName }
    $total = ($s.CounterSamples | Where-Object { $_.InstanceName -eq '_total' }).CookedValue
    $libre = ($s.CounterSamples | Where-Object { $_.Path -like '*available mbytes' }).CookedValue
    $usada = [math]::Round($ramTotal - $libre / 1024, 1)
    $gpu = (& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null) -split ',\s*'

    [Console]::SetCursorPosition(0, 0)
    Write-Host ("{0,-60}{1,36}" -f "USO DE RECURSOS DEL SISTEMA (evidencia de ejecucion)", (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor Cyan
    Write-Host ("CPU: {0}" -f $cpu).PadRight(96)
    Write-Host ("".PadRight(96, [char]0x2500)) -ForegroundColor DarkGray
    Write-Host ("USO POR NUCLEO LOGICO (12 hilos)".PadRight(60)) -ForegroundColor Yellow
    foreach ($n in $nucleos) {
        $p = [math]::Min(100, [math]::Round($n.CookedValue))
        $llenos = [int][math]::Round($p / 100 * 50)
        $color = if ($p -ge 70) { 'Red' } elseif ($p -ge 30) { 'Yellow' } else { 'Green' }
        Write-Host ("  Nucleo {0,2}  " -f $n.InstanceName) -NoNewline
        Write-Host ($bloque * $llenos) -NoNewline -ForegroundColor $color
        Write-Host (($vacio * (50 - $llenos)) + ("  {0,3} %" -f $p)).PadRight(40) -ForegroundColor DarkGray
    }
    Write-Host ("".PadRight(96, [char]0x2500)) -ForegroundColor DarkGray
    Write-Host ("  CPU TOTAL : {0,5:N1} %" -f $total).PadRight(96) -ForegroundColor White
    Write-Host ("  MEMORIA   : {0,5:N1} GB usados de {1} GB" -f $usada, $ramTotal).PadRight(96) -ForegroundColor White
    Write-Host ("".PadRight(96, [char]0x2500)) -ForegroundColor DarkGray
    if ($gpu.Count -ge 4) {
        Write-Host ("  GPU       : {0}" -f $gpu[0]).PadRight(96) -ForegroundColor White
        Write-Host ("  USO GPU   : {0,3} %          MEMORIA GPU: {1} / {2} MiB" -f $gpu[1], $gpu[2], $gpu[3]).PadRight(96) -ForegroundColor White
    }
}
