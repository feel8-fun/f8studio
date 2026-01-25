param(
  [switch]$Check,
  [switch]$ChangedOnly,
  [switch]$StagedOnly,
  [string]$ClangFormat = ""
)

$ErrorActionPreference = "Stop"

function Resolve-ClangFormat([string]$value) {
  if ($value -and (Test-Path $value)) { return (Resolve-Path $value).Path }

  if ($value) {
    $byName = Get-Command $value -ErrorAction SilentlyContinue
    if ($byName) { return $byName.Source }
  }

  $cmd = Get-Command "clang-format" -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $candidates = @()
  if ($env:LLVM_HOME) {
    $candidates += (Join-Path $env:LLVM_HOME "bin\\clang-format.exe")
  }
  if ($env:VCToolsInstallDir) {
    $candidates += (Join-Path $env:VCToolsInstallDir "Llvm\\x64\\bin\\clang-format.exe")
    $candidates += (Join-Path $env:VCToolsInstallDir "Llvm\\bin\\clang-format.exe")
  }
  if ($env:ProgramFiles) {
    $candidates += (Join-Path $env:ProgramFiles "LLVM\\bin\\clang-format.exe")
  }
  if (${env:ProgramFiles(x86)}) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\\Installer\\vswhere.exe"
    if (Test-Path $vswhere) {
      $vsPath = & $vswhere -latest -products * -property installationPath 2>$null
      if ($vsPath) {
        $candidates += (Join-Path $vsPath "VC\\Tools\\Llvm\\x64\\bin\\clang-format.exe")
      }
    }
  }

  foreach ($p in $candidates) {
    if (Test-Path $p) { return (Resolve-Path $p).Path }
  }

  # Last resort: scan common VS install paths (slower).
  foreach ($root in @("C:\\Program Files\\Microsoft Visual Studio", "C:\\Program Files (x86)\\Microsoft Visual Studio")) {
    if (-not (Test-Path $root)) { continue }
    $hit = Get-ChildItem $root -Recurse -Filter clang-format.exe -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -match "\\\\VC\\\\Tools\\\\Llvm\\\\" } |
      Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }

  Write-Error "Missing clang-format. Install LLVM (clang-format) or VS 'LLVM/Clang tools', or pass -ClangFormat <path>."
  return $null
}

Set-Location (Split-Path -Parent $PSScriptRoot)
$clang = Resolve-ClangFormat $ClangFormat
Write-Host "[clang-format] using: $clang"

$extensions = @("*.c", "*.cc", "*.cpp", "*.cxx", "*.h", "*.hh", "*.hpp", "*.hxx")

if ($ChangedOnly -and $StagedOnly) {
  Write-Error "Use -ChangedOnly or -StagedOnly, not both."
}

if ($StagedOnly) {
  $files = git diff --name-only --cached --diff-filter=ACMRTUXB
} elseif ($ChangedOnly) {
  $files = @()
  $files += git diff --name-only --diff-filter=ACMRTUXB
  $files += git diff --name-only --cached --diff-filter=ACMRTUXB
  $files = $files | Sort-Object -Unique
} else {
  $files = git ls-files @extensions
}

$files = $files | Where-Object {
  $p = $_.Replace("\\", "/")
  $extensions | Where-Object { $p -like $_ }
}

if (-not $files) {
  Write-Host "[clang-format] no C/C++ files tracked by git"
  exit 0
}

# Exclude vendored / generated / legacy folders if needed.
$excludePrefixes = @(
  "build/",
  "ignore/",
  "packages/f8implayer/src/imgui_backends/"
)

$targetFiles = $files | Where-Object {
  $f = $_.Replace("\\", "/")
  -not ($excludePrefixes | Where-Object { $f.StartsWith($_) })
}

if (-not $targetFiles) {
  Write-Host "[clang-format] no files after exclusions"
  exit 0
}

if ($Check) {
  Write-Host "[clang-format] check: $($targetFiles.Count) files"
  & $clang -n -Werror --style=file @targetFiles
  exit $LASTEXITCODE
}

Write-Host "[clang-format] format: $($targetFiles.Count) files"
& $clang -i --style=file @targetFiles
exit $LASTEXITCODE
