[CmdletBinding()]
param(
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$HarnessVersion = '0.1.2',

    [string]$PythonLauncher = 'py'
)

$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'This installer is intended for Windows.'
}

$pycommuteVersion = '1.0.0'
$pycommuteSha256 = 'a45e8ed1198497030e3b311ebd45908c075903c9ab1ba29ba370a96d60e1b7c2'
$workName = 'quanllm-harness-install-' + [Guid]::NewGuid().ToString('N')
$work = Join-Path ([System.IO.Path]::GetTempPath()) $workName
New-Item -ItemType Directory -Path $work | Out-Null

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$FailureMessage
    )

    & $PythonLauncher @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

try {
    Write-Host 'Checking Python...'
    Invoke-Python -Arguments @('--version') -FailureMessage 'Python is unavailable.'

    Write-Host "Downloading pycommute $pycommuteVersion source with pip..."
    Invoke-Python -Arguments @(
        '-m', 'pip', 'download', "pycommute==$pycommuteVersion",
        '--no-deps', '--no-cache-dir', '--no-binary=pycommute',
        '--dest', $work
    ) -FailureMessage 'Failed to download the pycommute source archive with pip.'

    $archives = @(
        Get-ChildItem -Path $work -File -Filter "pycommute-$pycommuteVersion*.tar.gz"
    )
    if ($archives.Count -ne 1) {
        throw "Expected exactly one pycommute source archive, found $($archives.Count)."
    }

    $archive = $archives[0].FullName
    $actualHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $pycommuteSha256) {
        throw "pycommute archive hash mismatch. Actual hash: $actualHash"
    }

    Write-Host 'Extracting pycommute source...'
    tar.exe -xzf $archive -C $work
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to extract the pycommute source archive.'
    }

    $sourceDirectory = Get-ChildItem -Path $work -Directory |
        Where-Object { $_.Name -eq "pycommute-$pycommuteVersion" } |
        Select-Object -First 1
    if ($null -eq $sourceDirectory) {
        throw "Extracted pycommute source directory was not found in $work."
    }

    $sourceRoot = $sourceDirectory.FullName
    $header = Join-Path $sourceRoot 'src\libcommute\libcommute\expression\monomial.hpp'
    if (-not (Test-Path -LiteralPath $header -PathType Leaf)) {
        throw "monomial.hpp was not found: $header"
    }

    Write-Host 'Applying the MSVC comparator compatibility patch...'
    $headerSource = [System.IO.File]::ReadAllText($header)
    $oldComparator = 'std::greater<generator_type const&>()'
    $newComparator = 'std::greater<>()'
    if (-not $headerSource.Contains($oldComparator)) {
        throw 'The expected libcommute comparator expression was not found.'
    }
    [System.IO.File]::WriteAllText(
        $header,
        $headerSource.Replace($oldComparator, $newComparator)
    )

    $setupPath = Join-Path $sourceRoot 'setup.py'
    if (-not (Test-Path -LiteralPath $setupPath -PathType Leaf)) {
        throw "setup.py was not found: $setupPath"
    }

    Write-Host 'Applying the Windows extension-name compatibility patch...'
    $setupSource = [System.IO.File]::ReadAllText($setupPath)
    $moduleReplacements = [ordered]@{
        '"pycommute/expression"' = '"pycommute.expression"'
        '"pycommute/loperator"' = '"pycommute.loperator"'
    }
    foreach ($oldModuleName in $moduleReplacements.Keys) {
        if (-not $setupSource.Contains($oldModuleName)) {
            throw "The expected module declaration was not found: $oldModuleName"
        }
        $setupSource = $setupSource.Replace(
            $oldModuleName,
            $moduleReplacements[$oldModuleName]
        )
    }
    [System.IO.File]::WriteAllText($setupPath, $setupSource)

    Write-Host 'Installing Python build requirements...'
    Invoke-Python -Arguments @(
        '-m', 'pip', 'install', '--upgrade', '--only-binary=:all:',
        'setuptools', 'wheel', 'packaging', 'pybind11'
    ) -FailureMessage 'Failed to install the Python build requirements.'

    $previousCl = [System.Environment]::GetEnvironmentVariable('CL', 'Process')
    [System.Environment]::SetEnvironmentVariable('CL', '/Zc:__cplusplus', 'Process')
    try {
        Write-Host 'Building and installing the patched pycommute package...'
        Invoke-Python -Arguments @(
            '-m', 'pip', 'install', '--no-build-isolation', '--no-cache-dir', $sourceRoot
        ) -FailureMessage 'Failed to build and install the patched pycommute package.'
    }
    finally {
        [System.Environment]::SetEnvironmentVariable('CL', $previousCl, 'Process')
    }

    Write-Host "Installing QuanLLM Harness $HarnessVersion..."
    Invoke-Python -Arguments @(
        '-m', 'pip', 'install', "quanllm-harness==$HarnessVersion"
    ) -FailureMessage 'Failed to install QuanLLM Harness.'

    Write-Host 'Verifying the installation...'
    $verification = @'
from importlib.metadata import version
from pycommute.expression import c, c_dag

result = c(0) * c_dag(0) + c_dag(0) * c(0)
if str(result) != "1":
    raise RuntimeError(f"Unexpected fermion anticommutator: {result}")

print("pycommute", version("pycommute"))
print("quanllm-harness", version("quanllm-harness"))
print("fermion anticommutator:", result)
'@
    $verificationPath = Join-Path $work 'verify_installation.py'
    [System.IO.File]::WriteAllText($verificationPath, $verification)
    Invoke-Python -Arguments @($verificationPath) -FailureMessage 'Installation verification failed.'

    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue

    Write-Host ''
    Write-Host 'QuanLLM Harness installation completed successfully.' -ForegroundColor Green
}
catch {
    Write-Host ("ERROR: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host "Temporary files were kept for diagnosis: $work"
    exit 1
}
