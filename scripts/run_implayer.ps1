param(
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

$conanEnvCandidates = @(
  (Join-Path $root "build\\build\\generators\\conanrun.ps1"),
  (Join-Path $root "build\\generators\\conanrun.ps1")
)

foreach ($p in $conanEnvCandidates) {
  if (Test-Path $p) {
    . $p
    break
  }
}

$exeCandidates = @(
  (Join-Path $root "build\\build\\bin\\f8implayer_service.exe"),
  (Join-Path $root "build\\bin\\f8implayer_service.exe"),
  "f8implayer_service.exe"
)

$exe = $null
foreach ($p in $exeCandidates) {
  if ($p -is [string] -and (Test-Path $p)) {
    $exe = $p
    break
  }
}

if (-not $exe) {
  throw "f8implayer_service.exe not found. Build first (e.g. conan install + cmake --build)."
}

& $exe @Args
exit $LASTEXITCODE

