# 服务器代码上传（scp，不用 U 盘）
# 用法:
#   .\deploy_pack\upload_to_server.ps1
#   .\deploy_pack\upload_to_server.ps1 -Server "root@120.196.88.140" -Port 12222 -RemoteDir "/root/elephant_cloud"

param(
    [string]$Server = "root@120.196.88.140",
    [int]$Port = 12222,
    [string]$RemoteDir = "/root/elephant_cloud"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$files = @(
    "cloud_server.py",
    "cloud_inference.py",
    "cloud_render.py",
    "elephant_clip_recorder.py",
    "video_tracker_yolo.py",
    "predict.py",
    "classifier.py",
    "elephant_net.py",
    "paths.py",
    "allowed_elephants.json",
    "class_names.json",
    "cloud_clip_env.sh",
    "start_cloud_server_linux.sh",
    "start_cloud_server_cpu.sh",
    "start_cloud_server_low_vram.sh",
    "uovision_camera.py",
    "uovision_registry.py",
    "uovision_open_api.py",
    "uovision_video_pipeline.py",
    "uovision_cameras.json",
    "uovision_config.example.sh",
    "setup_uovision_ir.sh"
)

Write-Host "=== 上传到 ${Server}:${RemoteDir} (端口 ${Port}) ==="
foreach ($rel in $files) {
    $local = Join-Path $ProjectRoot $rel
    if (-not (Test-Path $local)) {
        Write-Warning "跳过（不存在）: $rel"
        continue
    }
    Write-Host "  -> $rel"
    & scp -P $Port $local "${Server}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) {
        throw "scp 失败: $rel"
    }
}

Write-Host ""
Write-Host "上传完成。"
Write-Host "SSH 登录并重启:"
Write-Host "  ssh -p $Port $Server"
Write-Host "  cd $RemoteDir && bash start_cloud_server_linux.sh"
