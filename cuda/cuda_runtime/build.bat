@echo off
REM Build script for libptxrt on Windows

setlocal enabledelayedexpansion

echo ==========================================
echo Building PTX Runtime Library (libptxrt)
echo ==========================================
echo.

REM Get the directory of this script
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Create build directory
set BUILD_DIR=build
if exist "%BUILD_DIR%" (
    echo Cleaning existing build directory...
    rmdir /s /q "%BUILD_DIR%"
)

mkdir "%BUILD_DIR%"
cd "%BUILD_DIR%"

REM Configure
echo Configuring with CMake...
cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%SCRIPT_DIR%install"

if errorlevel 1 (
    echo CMake configuration failed!
    exit /b 1
)

REM Build
echo.
echo Building...
cmake --build . --config Release

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo ==========================================
echo Build completed successfully!
echo ==========================================
echo.
echo Libraries built:
echo   - Static: %BUILD_DIR%\Release\ptxrt.lib
echo   - Shared: %BUILD_DIR%\Release\ptxrt.dll
echo.
echo To install, run:
echo   cd %BUILD_DIR% ^&^& cmake --install .
echo.
echo To use in your CUDA programs:
echo   clang++ your_program.cu ^
echo     --cuda-path=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0 ^
echo     --cuda-gpu-arch=sm_61 ^
echo     -I%SCRIPT_DIR% ^
echo     -L%BUILD_DIR%\Release ^
echo     -lptxrt ^
echo     -o your_program.exe
echo.

endlocal
