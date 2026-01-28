param(
  [switch]$Build,
  [switch]$UseGstreamer,
  [string]$Codec = "VP8",
  [int]$Width = 640,
  [int]$Height = 360,
  [int]$Fps = 20,
  [int]$MinFrames = 3,
  [double]$Duration = 8.0,
  [double]$Timeout = 12.0,
  [string]$ServiceId = "webrtc_gateway",
  [string]$Ws = "ws://127.0.0.1:8765/ws",
  [string]$NatsUrl = "nats://127.0.0.1:4222"
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
  try {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
  } catch {
    return (Get-Location).Path
  }
}

$root = Resolve-RepoRoot
Push-Location $root
try {
  $buildRoot = Join-Path $root "build"
  $exe = Join-Path $buildRoot "bin\\f8webrtc_gateway_service.exe"

  if ($Build -or -not (Test-Path $exe)) {
    Write-Host "[test] building gateway..."
    & conan install . -s build_type=Release -o with_gst_webrtc=True -o with_tests=True --build=missing
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & cmake --preset dev
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & cmake --build --preset dev --target f8webrtc_gateway_service
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }

  # Ensure GStreamer plugin discovery works (GST_PLUGIN_PATH, GST_PLUGIN_SCANNER, PATH).
  $runEnv = Join-Path $buildRoot "generators\\conanrun.ps1"
  if (Test-Path $runEnv) {
    . $runEnv
  }

  $runner = @(
    "run",
    ".\scripts\webrtc_gateway_video_test.py",
    "--start-gateway",
    "--gateway-exe", $exe,
    "--service-id", $ServiceId,
    "--ws", $Ws,
    "--nats-url", $NatsUrl,
    "--codec", $Codec,
    "--width", $Width,
    "--height", $Height,
    "--fps", $Fps,
    "--min-frames", $MinFrames,
    "--duration", $Duration,
    "--timeout", $Timeout
  )
  if ($UseGstreamer) {
    $runner += "--video-use-gstreamer"
  }

  Write-Host "[test] uv $($runner -join ' ')"
  & uv @runner
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
