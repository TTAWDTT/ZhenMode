$ErrorActionPreference = 'Stop'
$stageRoot = 'D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/neutral-shear-paired-be647-attempt02'
$historyRoot = 'D:/Github/ocean-solver/outputs/vertical-mean-round3-20261012/independent-neutral-shear-native'
$reviewRoot = Join-Path $stageRoot 'review/report-frame'
$draftPath = Join-Path $stageRoot 'results-draft03.md'
$draftLines = Get-Content -LiteralPath $draftPath
$gridCases = @{'8 x 12 x 17'='00-main'; '16 x 24 x 33'='01-main'; '32 x 48 x 65'='02-main'}
$records = @{}
$identities = @()
foreach ($caseName in '00-main','01-main','02-main') {
    foreach ($kind in 'historical','new') {
        $rootPath = if ($kind -eq 'historical') { $historyRoot } else { $stageRoot }
        $recordPath = Join-Path $rootPath "$caseName/science-verdict01.json"
        $records["$kind/$caseName"] = Get-Content -Raw -LiteralPath $recordPath
        $parsed = $records["$kind/$caseName"] | ConvertFrom-Json
        $identities += [ordered]@{kind=$kind;case=$caseName;path=$recordPath;sha256=(Get-FileHash -LiteralPath $recordPath -Algorithm SHA256).Hash.ToLowerInvariant();status=$parsed.status}
    }
}
function Compare-RecordToken ($caseName,$kind,$metric,$printed,$lineNumber) {
    $pattern = '"' + [regex]::Escape($metric) + '"\s*:\s*([-+0-9.eE]+)'
    $match = [regex]::Match($records["$kind/$caseName"],$pattern)
    if (-not $match.Success) { throw "Missing literal metric: $kind/$caseName/$metric" }
    $rawToken = $match.Groups[1].Value
    $numberStyle = [System.Globalization.NumberStyles]::Float
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $sameDecimal = [decimal]::Parse($printed,$numberStyle,$culture) -eq [decimal]::Parse($rawToken,$numberStyle,$culture)
    [ordered]@{line=$lineNumber;case=$caseName;kind=$kind;metric=$metric;printed=$printed;source_json_token=$rawToken;exact_decimal_equal=$sameDecimal}
}
$checks = @()
for ($lineIndex=108; $lineIndex -le 125; $lineIndex++) {
    $cells = $draftLines[$lineIndex].Split('|') | ForEach-Object {$_.Trim()}
    $caseName = $gridCases[$cells[1]]
    $metric = ($cells[2].Split(',')[0]).Replace(' RMS','_RMS')
    $checks += Compare-RecordToken $caseName 'historical' $metric $cells[3] ($lineIndex+1)
    $checks += Compare-RecordToken $caseName 'new' $metric $cells[4] ($lineIndex+1)
}
for ($lineIndex=131; $lineIndex -le 133; $lineIndex++) {
    $cells = $draftLines[$lineIndex].Split('|') | ForEach-Object {$_.Trim()}
    $caseName = '{0:D2}-main' -f [int]$cells[1]
    $checks += Compare-RecordToken $caseName 'historical' 'T_normalized_L1' $cells[2] ($lineIndex+1)
    $checks += Compare-RecordToken $caseName 'historical' 'S_normalized_L1' $cells[3] ($lineIndex+1)
    $checks += Compare-RecordToken $caseName 'new' 'T_normalized_L1' $cells[4] ($lineIndex+1)
    $checks += Compare-RecordToken $caseName 'new' 'S_normalized_L1' $cells[5] ($lineIndex+1)
}
$failed = @($checks | Where-Object {-not $_.exact_decimal_equal})
$record = [ordered]@{
    status=if ($failed.Count -eq 0) {'PASS_LITERAL_RECORD_TRANSCRIPTION_ONLY'} else {'FAIL_LITERAL_RECORD_TRANSCRIPTION'}
    report_sha256=(Get-FileHash -LiteralPath $draftPath -Algorithm SHA256).Hash.ToLowerInvariant()
    scope='Literal decimal transcription of saved 3 historical and 3 new science JSON metrics into RMS and normalized-L1 comparison tables only; no scientific computation.'
    check_count=$checks.Count;failed_count=$failed.Count;source_records=$identities;checks=$checks
    npz_decoded=0;model_calls=0;observer_calls=0;physics_metrics_computed=0;ratios_computed=0;source_ci_repeated=$false;controls_3_4_assessed=$false;final_summary_assessed=$false
}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reviewRoot 'draft03-number-check01.json') -Encoding utf8
[pscustomobject]@{status=$record.status;report_sha256=$record.report_sha256;check_count=$record.check_count;failed_count=$record.failed_count} | ConvertTo-Json
if ($failed.Count) { throw 'Saved metric transcription mismatch' }
