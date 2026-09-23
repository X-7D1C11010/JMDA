param(
    [string]$PythonExe = "D:\Anaconda\envs\pytorch\python.exe",
    [string]$DataRoot = "D:\Code\JMDA-Net\Data",
    [string]$TargetRoot = "all",
    # The detected local RTX 4060 Ti has 8 GiB; 8 is the verified safe value.
    [int]$BatchSize = 8,
    [int]$Epochs = 100,
    [int]$NumIterations = 5,
    [string[]]$TransportModes = @(
        "legacy_row_softmax",
        "row_softmax",
        "sinkhorn"
    )
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Construct the Chinese directory name from Unicode code points so this file
# also works in Windows PowerShell 5.1, which may decode UTF-8 files as ANSI.
$sunnyDomainName = ([string][char]0x6674) + ([string][char]0x5929)
$sourceRoot = Join-Path $DataRoot $sunnyDomainName
$entryPoint = Join-Path $scriptDir "module_ablation.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python interpreter not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Source domain not found: $sourceRoot"
}

foreach ($mode in $TransportModes) {
    Write-Host "============================================================"
    Write-Host "Running supervised transport ablation: $mode"
    Write-Host "source=$sourceRoot, target=$TargetRoot"
    Write-Host "============================================================"

    & $PythonExe $entryPoint `
        --ablation_mode full `
        --use_target_labels `
        --no_auto_ablation_hparams `
        --transport_mode $mode `
        --source_root $sourceRoot `
        --target_root $TargetRoot `
        --batch_size $BatchSize `
        --epochs $Epochs `
        --num_iterations $NumIterations

    if ($LASTEXITCODE -ne 0) {
        throw "Transport ablation failed for mode: $mode"
    }
}
