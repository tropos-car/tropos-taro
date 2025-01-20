@echo off
setlocal enabledelayedexpansion

:: Get location name, source folder and destination root folder from command-line arguments
set "locationName=%~1"
set "serialNumber=%~2"
set "sourceFolder=%~3"
set "destinationRoot=%~4"

:: Check if location name, source and destination folders are provided
if "%locationName%"=="" (
    echo Commandline argument is missing.
    exit /b 1
)
if "%sourceFolder%"=="" (
    echo Commandline argument is missing.
    exit /b 1
)
if "%serialNumber%"=="" (
    echo Commandline argument is missing.
    exit /b 1
)
if "%destinationRoot%"=="" (
    echo Commandline argument is missing.
    exit /b 1
)

:: Get the current date and time in the format YYYYMMDDHH24
for /f "delims=" %%a in ('powershell Get-Date -Format "yyyyMMddHHmm"') do (
    set "currentDate=%%a"
)

:: Move ASi-16 images
for /r %sourceFolder%\asi_%serialNumber% %%F in (*.jpg) do (
    set "fileName=%%~nF"
    set "fileDate=!fileName:~0,12!"
    set "year=!fileDate:~0,4!"
    set "month=!fileDate:~4,2!"
    set "day=!fileDate:~6,2!"

    if "!fileDate!" neq "!currentDate:~0,12!" (
        set "destinationFolder=%destinationRoot%\l0\!year!\!month!\!day!"
        if not exist "!destinationFolder!" (
            mkdir "!destinationFolder!"
        )
        set "newFileName=asi16_!locationName!_%%~nxF"
        move "%%F" "!destinationFolder!\!newFileName!"
    )
)

:: Move ASi-16 evaluation images
for /r %sourceFolder%\eval\images\asi_%serialNumber% %%F in (*.png *.bmp) do (
    set "fileName=%%~nF"
    set "fileDate=!fileName:~0,12!"
    set "year=!fileDate:~0,4!"
    set "month=!fileDate:~4,2!"
    set "day=!fileDate:~6,2!"

    if "!fileDate!" neq "!currentDate:~0,12!" (
        set "destinationFolder=%destinationRoot%\l1a\images\!year!\!month!\!day!"
        if not exist "!destinationFolder!" (
            mkdir "!destinationFolder!"
        )
        set "newFileName=asi16_!locationName!_%%~nxF"
        move "%%F" "!destinationFolder!\!newFileName!"
    )
)


