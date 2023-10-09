@echo off
setlocal enabledelayedexpansion

:: Get source folder and destination root folder from command-line arguments
set "sourceFolder=%~1"
set "destinationRoot=%~2"

:: Check if both source and destination folders are provided
if "%sourceFolder%"=="" (
    echo Source folder path is missing.
    exit /b 1
)

if "%destinationRoot%"=="" (
    echo Destination root folder path is missing.
    exit /b 1
)

:: Get the current date and time in the format YYYYMMDDHH24
for /f "delims=" %%a in ('powershell Get-Date -Format "yyyyMMddHH"') do (
    set "currentDate=%%a"
)


for %%F in (%sourceFolder%\%currentDate:~0,4%\*.WSD) do (
    set "fileName=%%~nF"
    set "fileDate=!fileName:~0,8!"
    set "year=!fileDate:~0,4!"
    set "month=!fileDate:~4,2!"
    
    if "!fileDate!" neq "!currentDate:~0,8!" (
        set "destinationFolder=%destinationRoot%\!year!\!month!"
        if not exist "!destinationFolder!" (
            mkdir "!destinationFolder!"
        )
        
        move "%%F" "!destinationFolder!\!fileName!.WSD"
    )
)

for %%F in (%sourceFolder%\%currentDate:~0,4%\*.CSV) do (
    set "fileName=%%~nF"
    set "fileDate=!fileName:~0,10!"
    set "year=!fileDate:~0,4!"
    set "month=!fileDate:~4,2!"
    set "day=!fileDate:~6,2!"
    
    if "!fileDate!" neq "!currentDate!" (
        set "destinationFolder=%destinationRoot%\!year!\!month!\!day!"
        if not exist "!destinationFolder!" (
            mkdir "!destinationFolder!"
        )
    
        move "%%F" "!destinationFolder!\!fileName!.CSV"
    )
)
