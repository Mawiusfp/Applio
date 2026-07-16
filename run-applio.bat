@echo off

@REM set CUDA_LAUNCH_BLOCKING=1
@REM set TORCH_USE_CUDA_DSA=1

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo Applio does not require administrator permissions and should be run as a regular user.
    echo.
    pause
    exit /b 1
)

setlocal
for %%F in ("%~dp0.") do set "folder_name=%%~nF"

title %folder_name%

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

env\python.exe app.py
echo.
pause
