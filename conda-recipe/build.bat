@echo off
REM Helper script to build samurai-python conda package locally on Windows
REM
REM Usage:
REM   build.bat              # Build for current platform
REM   build.bat test         # Build and test
REM   build.bat install      # Build and install locally

setlocal enabledelayedexpansion

echo ============================================
echo Samurai Python Conda Build Helper (Windows)
echo ============================================
echo.

REM Get the directory of this script
set SCRIPT_DIR=%~dp0
set PYTHON_DIR=%SCRIPT_DIR%..

REM Check if samurai C++ package is installed
echo Checking for samurai C++ package...
conda list samurai 2>nul | findstr /C:"samurai" >nul
if errorlevel 1 (
    echo Error: samurai C++ package not found!
    echo Please install it first:
    echo   conda install -c conda-forge samurai
    echo Or build it from source before building the Python bindings.
    exit /b 1
)
echo Found samurai C++ package
echo.

REM Build the package
echo Building samurai-python conda package...
cd /d "%PYTHON_DIR%"

conda build --no-anaconda-upload conda-recipe\

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo Build successful!
echo.

REM Get the output path
for /f "delims=" %%i in ('conda build --output conda-recipe\ 2^>nul') do set OUTPUT_PATH=%%i
echo Package location: %OUTPUT_PATH%
echo.

REM Handle arguments
if "%1"=="test" (
    echo Testing package...
    echo Tests completed during build
)

if "%1"=="install" (
    echo Installing package locally...
    conda install --use-local samurai-python
    echo Package installed!
    echo.
    echo Test the installation:
    echo   python -c "import samurai_python; print(samurai_python.__version__)"
)

if "%1"=="upload" (
    echo Preparing package for upload...
    echo Note: Manual upload to anaconda.org is required
    echo.
    echo To upload to anaconda.org:
    echo   anaconda upload %OUTPUT_PATH%
    echo.
    echo Or create a PR to conda-forge:
    echo   https://github.com/conda-forge/staged-recipes
)

echo.
echo ============================================
echo Build Complete
echo ============================================
