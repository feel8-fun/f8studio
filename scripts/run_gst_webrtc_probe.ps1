param(
  [string]$BuildDir = "build",
  [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildDirFull = (Join-Path $RepoRoot $BuildDir)
New-Item -ItemType Directory -Force -Path $BuildDirFull | Out-Null

Write-Host "[1/4] conan export local gstreamer recipes"
conan export (Join-Path $RepoRoot "conan_recipes/gst_plugins_base_recipe")
conan export (Join-Path $RepoRoot "conan_recipes/gst_plugins_good_recipe")
conan export (Join-Path $RepoRoot "conan_recipes/gst_plugins_bad_recipe")

Write-Host "[2/4] conan install (enable f8Build:with_gst_webrtc=True)"
conan install $RepoRoot -s build_type=$BuildType -o with_gst_webrtc=True -o with_tests=True --build=missing

Write-Host "[3/4] cmake configure + build f8gst_probe"
cmake --preset dev
cmake --build --preset dev --target f8gst_probe

Write-Host "[4/4] run f8gst_probe (with conan runenv)"
. (Join-Path $BuildDirFull "generators/conanrun.ps1")
& (Join-Path $BuildDirFull "bin/f8gst_probe.exe")
. (Join-Path $BuildDirFull "generators/deactivate_conanrun.ps1")
