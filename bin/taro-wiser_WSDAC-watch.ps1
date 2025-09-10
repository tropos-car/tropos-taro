param (
    # -----------------------------------------------------------------
    # 1️⃣ Pfad zum Ordner, in dem nach der Datei gesucht werden soll
    # -----------------------------------------------------------------
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateScript({ Test-Path $_ -PathType Container })]
    [string]$BaseWatchFolder,

    # -----------------------------------------------------------------
    # 2️⃣ (Optional) Pfad zum Programm, das neu gestartet werden soll.
    #    Wenn nichts angegeben wird, wird standardmäßig Notepad benutzt.
    # -----------------------------------------------------------------
    [Parameter(Mandatory=$false, Position=1)]
    [ValidateScript({ Test-Path $_ -PathType Leaf })]
    [string]$ProgramPath = "C:\EKO\MS-713\WSDac_5021.exe"
)

$today = Get-Date -Format "yyyy\\MM\\dd"
# Vollständiger Pfad, in dem wir suchen
$WatchFolder = Join-Path -Path $BaseWatchFolder -ChildPath $today

# Muster für den Dateinamen (inkl. Zeitstempel)
# Beispiel:  YYYY-MM-DDTHH_taro-wiser_actris-lacros.CSV
$filePattern = "????-??-??T??_taro-wiser_*.CSV"   # passt auf jede Erweiterung



# ------------------------------------------------------------
# 2️⃣ Hilfsfunktion: Zeitstempel aus Dateinamen extrahieren
# ------------------------------------------------------------
function Get-TimestampFromName {
    param([string]$Name)

    # Erwartetes Format: YYYYMMDDTHH_.<ext>
    # Regex zerlegt den Namen und gibt das Datum zurück
    if ($Name -match '(\d{4})-(\d{2})-(\d{2})T(\d{2})') {
        $year = $matches[1]   # YYYY
        $month = $matches[2]   # MM
        $day = $matches[3]   # DD
        $hour = $matches[4]   # HH
        $minute = "00"
        $second = "00"

        try {
            return [datetime]::ParseExact(
                "$year$month$day$hour$minute$second",
                "yyyyMMddHHmmss",
                $null
            )
        } catch {
            return $null
        }
    }
    return $null
}

# ------------------------------------------------------------
# 3️⃣ Hauptlogik
# ------------------------------------------------------------
# Alle Dateien finden, die dem Muster entsprechen
$candidateFiles = Get-ChildItem -Path $watchFolder -Filter $filePattern -File

if (-not $candidateFiles) {
    Write-Host "Keine Datei mit Zeitstempel Muster im Ordner gefunden."
    $needsRestart = $true
} else {
    # Für jede Datei den eingebetteten Zeitstempel auslesen
    $filesWithTimestamp = foreach ($f in $candidateFiles) {
        $ts = Get-TimestampFromName -Name $f.Name
        if ($ts) { [pscustomobject]@{File=$f; Timestamp=$ts} }
    }

    if (-not $filesWithTimestamp) {
        Write-Host "Gefundene Dateien besitzen keinen gültigen Zeitstempel im Namen."
        $needsRestart = $true
    } else {
        # Die aktuellste (nach Zeitstempel) Datei auswählen
        $latest = $filesWithTimestamp |
                  Sort-Object -Property Timestamp -Descending |
                  Select-Object -First 1

        $now = Get-Date
        $ageInMinutes = ($now - $latest.Timestamp).TotalMinutes

        if ($ageInMinutes -le 120) {
            #Write-Host "Neueste Datei '$($latest.File.Name)' wurde vor '$([math]::Round($ageInMinutes,1))' Minuten erstellt – kein Neustart nötig."
            Write-Host "no need restart"
            $needsRestart = $false
        } else {
            #Write-Host "Neueste Datei '$($latest.File.Name)' ist '$([math]::Round($ageInMinutes,1))' Minuten alt (>60min)."
            $needsRestart = $true
            Write-Host "need restart"
        }
    }
}

# ------------------------------------------------------------
# 4️⃣ Neustart des Programms, falls nötig
# ------------------------------------------------------------
if ($needsRestart) {
    Write-Host "Starte Programm neu..."

    # Laufende Instanz beenden (falls vorhanden)
    $procName = [System.IO.Path]::GetFileNameWithoutExtension($programPath)
    Get-Process -Name $procName -ErrorAction SilentlyContinue | Stop-Process -Force

    # Neu starten
    Start-Process -FilePath $programPath
    Write-Host "Programm wurde neu gestartet."
}