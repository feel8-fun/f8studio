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

$explicitExe = $env:F8AUDIOCAP_EXE
if ($explicitExe) {
  if (-not (Split-Path $explicitExe -IsAbsolute)) {
    $explicitExe = Join-Path $root $explicitExe
  }
  $explicitExe = (Resolve-Path $explicitExe -ErrorAction Stop).Path
}

$exeItems = @()

if ($explicitExe) {
  if (-not (Test-Path $explicitExe)) {
    throw "F8AUDIOCAP_EXE points to missing file: $explicitExe"
  }
  $exeItems += Get-Item $explicitExe
} else {
  $common = @(
    (Join-Path $root "build\\bin\\f8audiocap_service.exe")
  )

  foreach ($p in $common) {
    if (Test-Path $p) { $exeItems += Get-Item $p }
  }

  if ($exeItems.Count -eq 0 -and (Test-Path (Join-Path $root "build"))) {
    $exeItems += Get-ChildItem -Path (Join-Path $root "build") -Recurse -Filter "f8audiocap_service.exe" -ErrorAction SilentlyContinue
  }
}

if ($exeItems.Count -eq 0) {
  throw "f8audiocap_service.exe not found. Build first (e.g. conan install + cmake --build)."
}

$exeItems =
  $exeItems |
  Sort-Object `
    @{ Expression = { if ($_.FullName -match "\\\\build\\\\bin\\\\") { 0 } else { 1 } }; Ascending = $true }, `
    @{ Expression = { $_.LastWriteTimeUtc }; Descending = $true }

$exe = $exeItems[0].FullName

Write-Verbose "repoRoot: $root"
Write-Verbose "exe:      $exe"

if ($PrintOnly) {
  Write-Output $exe
  exit 0
}

Push-Location $root
try {
  & $exe @Args
  exit $LASTEXITCODE
} finally {
  Pop-Location
}

