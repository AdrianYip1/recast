# Runs the Python pipeline on a scene photo + an object photo, writing data/<Name>/.
# Then open the result in the engine (see the "foto.exe" launch targets in Visual Studio).
#
# Example (from the repo root, in Visual Studio's terminal):
#   .\test_images\run_test.ps1 -Scene test_images\empty_table.jpg -Object test_images\apple.jpg
#
# Only -Scene and -Object are required; everything else has a default that suits the
# apple/table test. Leave out -Base/-Top to click them yourself in a window instead.

param(
    [Parameter(Mandatory = $true)] [string] $Scene,    # photo of the empty scene (.jpg)
    [Parameter(Mandatory = $true)] [string] $Object,   # photo of the object (.jpg)
    [string] $Name,                                     # scene folder under data/ (default: <scene>_<object>)

    # scene camera
    [double] $CameraHeight = 1.2,                       # metres; sets the scale of the room
    [double] $SceneFocalPx = 0,                         # 0 = solve from lines; the table photo needs 470
    [string] $Lines = "",                               # hand-traced lines JSON; "" = detect automatically
    [switch] $Overwrite,                                # start scene.json fresh (drops boxes added by hand)
    [switch] $Detect,                                   # also find walls/objects with the models (needs .[models])

    # object
    [ValidateSet("card", "revolution", "box", "extrude", "inflate")] [string] $Mode = "revolution",
    [double] $Height = 0.08,                            # real height of the object in metres
    [string] $Mask = "",                                # ready-made cutout (white = object); "" = GrabCut
    [double] $ObjectFocalPx = 1500,                     # 1500 suits studio/product shots, ~500 a phone photo
    [double] $PitchDeg = -10,                           # how far the object photo looks down
    [double[]] $Base,                                   # u v of where the object touches the ground
    [double[]] $Top,                                    # u v of the top of the object's axis
    [int[]] $Rect                                       # x y w h box around the object (GrabCut only)
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo "python\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv missing: cd python; python -m venv .venv; .venv\Scripts\pip install -e "".[dev,lens]""" }
$Scene = (Resolve-Path $Scene).Path
$Object = (Resolve-Path $Object).Path
if (-not $Name) {
    $Name = ([IO.Path]::GetFileNameWithoutExtension($Scene) + "_" + [IO.Path]::GetFileNameWithoutExtension($Object)) -replace '[^A-Za-z0-9_-]', '_'
}

function Invoke-Step([string] $title, [string[]] $arguments) {
    Write-Host "`n=== $title ===" -ForegroundColor Cyan
    # Windows PowerShell treats any stderr line (library warnings) as an error under "Stop";
    # the exit code is what says whether the step worked
    $ErrorActionPreference = "Continue"
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
    & $py -m recast @arguments 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$title failed (exit $LASTEXITCODE)" }
}

# A. scene photo -> photo.jpg + camera
$a = @("scene", $Scene, "--name", $Name, "--camera-height", "$CameraHeight", "--no-lensfun")
if ($SceneFocalPx -gt 0) { $a += @("--focal-px", "$SceneFocalPx") }
if ($Lines) { $a += @("--lines", (Resolve-Path $Lines).Path) }
if ($Overwrite) { $a += "--overwrite" }
Invoke-Step "A. scene camera" $a

# A2. walls, ceiling and every object in the scene photo -> boxes the object can sit on / bump into
if ($Detect) { Invoke-Step "A2. detect walls + objects" @("detect", "--name", $Name) }

# B. object photo -> object.glb
$b = @("object", $Object, "--name", $Name, "--mode", $Mode, "--height", "$Height",
       "--focal-px", "$ObjectFocalPx", "--pitch-deg", "$PitchDeg", "--no-auto-lines", "--no-lensfun")
if ($Mask) { $b += @("--mask", (Resolve-Path $Mask).Path) }
if ($Rect) { $b += @("--rect") + ($Rect | ForEach-Object { "$_" }) }
if ($Base -and $Top) { $b += @("--base") + ($Base | ForEach-Object { "$_" }) + @("--top") + ($Top | ForEach-Object { "$_" }) + @("--no-ui") }
Invoke-Step "B. object model" $b

# C. lighting fit
Invoke-Step "C. lighting" @("light", "--name", $Name, "--python-gbuffer")

$out = Join-Path $repo "data\$Name"
Write-Host "`nDone: $out" -ForegroundColor Green
Write-Host "  object_check.jpg  green = cutout, red = 3D model outline"
Write-Host "  scene.json        open it in the engine:"
Write-Host "    out\build\x64-Debug\foto.exe ..\..\..\data\$Name\scene.json   (run from out\build\x64-Debug)"
