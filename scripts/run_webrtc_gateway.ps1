param(
  [switch]$PrintOnly,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
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

$explicitExe = $env:F8WEBRTC_GATEWAY_EXE
if ($explicitExe) {
  if (-not (Split-Path $explicitExe -IsAbsolute)) {
    $explicitExe = Join-Path $root $explicitExe
  }
  $explicitExe = (Resolve-Path $explicitExe -ErrorAction Stop).Path
}

$exeItems = @()
if ($explicitExe) {
  if (-not (Test-Path $explicitExe)) {
    throw "F8WEBRTC_GATEWAY_EXE points to missing file: $explicitExe"
  }
  $exeItems += Get-Item $explicitExe
} else {
  $common = @(
    (Join-Path $root "build\\bin\\f8webrtc_gateway_service.exe")
  )
  foreach ($p in $common) {
    if (Test-Path $p) { $exeItems += Get-Item $p }
  }
  if ($exeItems.Count -eq 0 -and (Test-Path (Join-Path $root "build"))) {
    $exeItems += Get-ChildItem -Path (Join-Path $root "build") -Recurse -Filter "f8webrtc_gateway_service.exe" -ErrorAction SilentlyContinue
  }
}

if ($exeItems.Count -eq 0) {
  throw "f8webrtc_gateway_service.exe not found. Build first (e.g. conan install + cmake --build)."
}

$exeItems =
  $exeItems |
  Sort-Object `
    @{ Expression = { if ($_.FullName -match "\\\\build\\\\bin\\\\") { 0 } else { 1 } }; Ascending = $true }, `
    @{ Expression = { $_.LastWriteTimeUtc }; Descending = $true }

$exe = $exeItems[0].FullName

$buildRoot = Split-Path (Split-Path $exe -Parent) -Parent
$runEnv = Join-Path $buildRoot "generators\\conanrun.ps1"
if (Test-Path $runEnv) {
  try {
    . $runEnv
  } catch {
    Write-Warning "Failed to load Conan runenv ($runEnv): $($_.Exception.Message)"
  }
} else {
  Write-Warning "Conan runenv not found: $runEnv (you may hit missing DLLs)"
}

Write-Verbose "repoRoot: $root"
Write-Verbose "exe:      $exe"
Write-Verbose "build:    $buildRoot"

if ($PrintOnly) {
  Write-Output $exe
  exit 0
}

Push-Location $root
try {
  $finalArgs = @()

  $hasServiceId = $false
  foreach ($a in $Args) {
    if ($a -eq "--service-id" -or $a -like "--service-id=*") { $hasServiceId = $true; break }
  }
  if (-not $hasServiceId) {
    $defaultServiceId = $env:F8WEBRTC_GATEWAY_SERVICE_ID
    if (-not $defaultServiceId) { $defaultServiceId = "webrtc_gateway" }
    $finalArgs += @("--service-id", $defaultServiceId)
  }

  $finalArgs += $Args

  & $exe @finalArgs
  if ($LASTEXITCODE -eq -1073741515) {
    Write-Warning "Process exited with STATUS_DLL_NOT_FOUND (0xC0000135). Run 'conan install .' then re-run this script so it can source build/generators/conanrun.ps1."
  }
  exit $LASTEXITCODE
} finally {
  Pop-Location
}
