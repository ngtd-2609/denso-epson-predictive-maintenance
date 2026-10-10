param([ValidateRange(1,5)][int]$Lop=1,[ValidateRange(1,100)][int]$SoMau=5,[int]$Seed=2026)
$ErrorActionPreference='Stop'
$pythonExe=Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if(!(Test-Path -LiteralPath $pythonExe)){throw 'Cai moi truong theo demo/README.md truoc khi chay.'}
$out=Join-Path $PSScriptRoot ('ket_qua_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
& $pythonExe (Join-Path $PSScriptRoot 'demo.py') --class-id $Lop --samples $SoMau --seed $Seed --out $out
if($LASTEXITCODE -ne 0){throw 'Demo failed. See error above.'}
Write-Output "Xem ket qua tai: $out"
