# Setup daily auto-commit task in Windows Task Scheduler
# Run this script ONCE as Administrator

$taskName = "GitHubAutoCommit"
$repoDir = "C:\Users\bookf\OneDrive\Desktop\New folder"
$pythonPath = (Get-Command python).Source

# Action: run auto_commit.py
$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "`"$repoDir\auto_commit.py`"" `
    -WorkingDirectory $repoDir

# Trigger: every day at 10:00, 14:00, 20:00 (3 times to be safe)
$trigger1 = New-ScheduledTaskTrigger -Daily -At 10:00AM
$trigger2 = New-ScheduledTaskTrigger -Daily -At 2:00PM
$trigger3 = New-ScheduledTaskTrigger -Daily -At 8:00PM

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Register the task
Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger1,$trigger2,$trigger3 `
    -Settings $settings `
    -Description "Daily auto-commit to GitHub for green squares" `
    -RunLevel Limited `
    -Force

Write-Host ""
Write-Host "Task '$taskName' created successfully!" -ForegroundColor Green
Write-Host "It will run daily at 10:00, 14:00, and 20:00"
Write-Host "Each run makes 3 commits + push to GitHub"
