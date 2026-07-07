@echo off
setlocal
cd /d D:\Code\JMDA-Net\Ablation
set PYTHONIOENCODING=utf-8
"D:\Anaconda\envs\pytorch\python.exe" "D:\Code\JMDA-Net\Ablation\run_all_requested_experiments.py" > "D:\Code\JMDA-Net\experiment_runs\launcher\scheduled_runner_stdout.log" 2> "D:\Code\JMDA-Net\experiment_runs\launcher\scheduled_runner_stderr.log"
exit /b %ERRORLEVEL%
