# 仅打包树莓派文件到 U 盘/移动硬盘（Pi 用硬盘拷贝，不用网络传代码）
# 用法: .\deploy_pack\copy_to_usb_for_pi.ps1 -UsbRoot "E:\elephant_pi"

param(
    [Parameter(Mandatory = $true)]
    [string]$UsbRoot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PiOut = $UsbRoot

New-Item -ItemType Directory -Force -Path $PiOut | Out-Null

function Copy-ProjectFile {
    param([string]$Rel, [string]$DestDir, [string]$DestName = "")
    $src = Join-Path $ProjectRoot $Rel
    if (-not (Test-Path $src)) {
        Write-Warning "跳过（不存在）: $Rel"
        return
    }
    $name = if ($DestName) { $DestName } else { Split-Path -Leaf $Rel }
    Copy-Item -Force $src (Join-Path $DestDir $name)
    Write-Host "  + $name"
}

Write-Host "=== 打包 Pi 文件 -> $PiOut ==="
Write-Host "（不含服务器代码；服务器请用 upload_to_server.ps1）"
Write-Host ""

$piFiles = @(
    @{ Rel = "pi_cloud_deploy/pi_cloud_client.py"; Name = "pi_cloud_client.py" },
    @{ Rel = "pi_cloud_deploy/pi_clip_recorder.py"; Name = "pi_clip_recorder.py" },
    @{ Rel = "pi_cloud_deploy/pi_cloud_config.sh"; Name = "pi_cloud_config.sh" },
    @{ Rel = "pi_cloud_deploy/run_pi_cloud_client.sh"; Name = "run_pi_cloud_client.sh" },
    @{ Rel = "pi_cloud_deploy/install_service.sh"; Name = "install_service.sh" },
    @{ Rel = "pi_cloud_deploy/pi_cloud_watchdog.sh"; Name = "pi_cloud_watchdog.sh" },
    @{ Rel = "pi_cloud_deploy/pi-cloud-client.service"; Name = "pi-cloud-client.service" },
    @{ Rel = "pi_cloud_deploy/pi-cloud-client-watchdog.service"; Name = "pi-cloud-client-watchdog.service" },
    @{ Rel = "pi_cloud_deploy/pi_cloud_client_config.example.sh"; Name = "pi_cloud_client_config.example.sh" },
    @{ Rel = "pi_cloud_deploy/requirements-pi-client.txt"; Name = "requirements-pi-client.txt" },
    @{ Rel = "pi_cloud_deploy/README.txt"; Name = "README.txt" },
    @{ Rel = "deploy_pack/pi/install_on_pi.sh"; Name = "install_on_pi.sh" }
)
foreach ($item in $piFiles) {
    Copy-ProjectFile $item.Rel $PiOut $item.Name
}

# 与 pi_cloud_deploy 保持同步的源文件
Copy-ProjectFile "pi_cloud_client.py" $PiOut
Copy-ProjectFile "pi_clip_recorder.py" $PiOut

Write-Host ""
Write-Host "完成！请将整个文件夹拷到 U 盘:"
Write-Host "  $PiOut"
Write-Host ""
Write-Host "Pi 上安装:"
Write-Host "  bash /media/pi/USB/elephant_pi/install_on_pi.sh /media/pi/USB/elephant_pi"
