$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$cohortName = 'course_completion_20261109-133839'
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and
    $_.CommandLine -like "*$cohortName*" -and
    ($_.CommandLine -like '*train_fraction.py*' -or $_.CommandLine -like '*run_course_completion.py*')
}
if ($running) {
    Write-Output 'The project queue or training is already running. No duplicate was started.'
    return
}
$runner = Join-Path $projectRoot 'final-report-work\methods\run_course_completion.py'
$cohort = Join-Path $projectRoot "final-report-work\methods\results\$cohortName"
if (-not (Test-Path -LiteralPath (Join-Path $cohort 'state.json'))) {
    throw 'Saved queue state was not found.'
}
$savedState = Get-Content -LiteralPath (Join-Path $cohort 'state.json') -Raw | ConvertFrom-Json
if ($savedState.status -eq 'experiments_complete') {
    Write-Output 'The experiment queue is already complete.'
    return
}
$pyLauncher = (Get-Command py.exe -ErrorAction Stop).Source
$launchArgs = @('-3.12', '-u', ('"' + $runner + '"'), ('"' + $cohort + '"'))
$resumed = Start-Process -FilePath $pyLauncher -ArgumentList $launchArgs `
    -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $projectRoot 'reports\resume_runner.log') `
    -RedirectStandardError (Join-Path $projectRoot 'reports\resume_runner_errors.log')
Write-Output "Resumed queue (launcher PID $($resumed.Id)). Training loads its most recent saved checkpoint."
