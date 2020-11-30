#!/bin/bash

$log = get-content D:\scripts\iis.log
foreach ($line in $log) { 
    if ($line -like "?????0, 0*" || "??????0, 0*") {
$line | out-file -FilePath "data_0_0.csv" -Append
    }
}