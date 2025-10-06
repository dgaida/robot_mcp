@echo off
REM Launch script for Robot Control GUI (Windows)
REM Usage: launch_gui.bat [options]

setlocal enabledelayedexpansion

REM Default values
set ROBOT=niryo
set SIMULATION=true
set MODEL=moonshotai/kimi-k2-instruct-0905
set SHARE=false

REM Colors (using ANSI escape codes)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Parse arguments
:parse_args
if "%~1"=="" goto check_deps
if /i "%~1"=="--robot" (
    set ROBOT=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--real" (
    set SIMULATION=false
    shift
    goto parse_args
)
if /i "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--share" (
    set SHARE=true
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help

echo %RED%Unknown option: %~1%NC%
goto show_help

:show_help
echo Usage: launch_gui.bat [options]
echo.
echo Options:
echo   --robot ROBOT       Robot type (niryo or widowx) [default: niryo]
echo   --real              Use real robot instead of simulation
echo   --model MODEL       Groq model to use
echo   --share             Create public Gradio link
echo   --help, -h          Show this help message
echo.
echo Examples:
echo   launch_gui.bat
echo   launch_gui.bat --robot widowx --real
echo   launch_gui.bat --model llama-3.1-8b-instant --share
exit /b 0

:check_deps
echo %BLUE%==================================%NC%
echo %BLUE%  Robot Control GUI Launcher%NC%
echo %BLUE%==================================%NC%
echo.

echo %YELLOW%Checking dependencies...%NC%

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%✗ Python not found%NC%
    echo %YELLOW%  Install Python 3.8 or higher%NC%
    exit /b 1
)
echo %GREEN%✓ Python found%NC%

REM Check packages
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo %RED%✗ gradio not installed%NC%
    echo %YELLOW%  Install with: pip install gradio%NC%
    exit /b 1
)
echo %GREEN%✓ gradio installed%NC%

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo %RED%✗ torch not installed%NC%
    echo %YELLOW%  Install with: pip install torch%NC%
    exit /b 1
)
echo %GREEN%✓ torch installed%NC%

python -c "import fastmcp" >nul 2>&1
if errorlevel 1 (
    echo %RED%✗ fastmcp not installed%NC%
    echo %YELLOW%  Install with: pip install fastmcp%NC%
    exit /b 1
)
echo %GREEN%✓ fastmcp installed%NC%

python -c "import groq" >nul 2>&1
if errorlevel 1 (
    echo %RED%✗ groq not installed%NC%
    echo %YELLOW%  Install with: pip install groq%NC%
    exit /b 1
)
echo %GREEN%✓ groq installed%NC%

echo.

:check_keys
echo %YELLOW%Checking API keys...%NC%

if not exist secrets.env (
    echo %RED%✗ secrets.env not found%NC%
    echo %YELLOW%  Creating template...%NC%
    (
        echo GROQ_API_KEY=your_groq_api_key_here
        echo ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
    ) > secrets.env
    echo %YELLOW%  Please edit secrets.env and add your API keys%NC%
    exit /b 1
)

REM Check for GROQ_API_KEY in secrets.env
findstr /C:"GROQ_API_KEY=" secrets.env >nul
if errorlevel 1 (
    echo %RED%✗ GROQ_API_KEY not found in secrets.env%NC%
    exit /b 1
)

findstr /C:"GROQ_API_KEY=your_groq_api_key_here" secrets.env >nul
if not errorlevel 1 (
    echo %RED%✗ Please set your GROQ_API_KEY in secrets.env%NC%
    exit /b 1
)

echo %GREEN%✓ GROQ_API_KEY found%NC%
echo.

:show_config
echo %BLUE%Configuration:%NC%
echo   Robot:      %ROBOT%
if "%SIMULATION%"=="true" (
    echo   Mode:       Simulation
) else (
    echo   Mode:       Real Robot
)
echo   Model:      %MODEL%
if "%SHARE%"=="true" (
    echo   Share:      Yes
) else (
    echo   Share:      No
)
echo.

:launch
echo %GREEN%🚀 Launching Robot Control GUI...%NC%
echo.

REM Build command
set CMD=python robot_gui/mcp_app.py --robot %ROBOT% --model %MODEL%

if "%SIMULATION%"=="false" (
    set CMD=!CMD! --no-simulation
)

if "%SHARE%"=="true" (
    set CMD=!CMD! --share
)

echo %YELLOW%Command: !CMD!%NC%
echo.

REM Execute
!CMD!

REM Cleanup on exit
:cleanup
echo.
echo %YELLOW%Shutting down...%NC%

REM Kill any remaining processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq main_server.py*" >nul 2>&1

echo %GREEN%✓ Cleanup complete%NC%

endlocal
