#!/bin/bash
set -euo pipefail

stats_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_file="${1:-$stats_dir/general.csv}"
output_file="${2:-$stats_dir/statistics.csv}"

if [[ ! -f "$input_file" ]]; then
    echo "Missing collected results: $input_file" >&2
    echo "Run the collect job before generating statistics.csv." >&2
    exit 1
fi

awk -F',' '
BEGIN {
    OFS = ","
}
NR == 1 {
    for (i = 1; i <= NF; i++) {
        column[$i] = i
    }
    required[1] = "Instance"
    required[2] = "BestObjective"
    required[3] = "MeanBestObjective"
    required[4] = "StdBestObjective"
    required[5] = "MeanRuntime"
    for (i = 1; i <= 5; i++) {
        if (!(required[i] in column)) {
            print "Missing column " required[i] " in general.csv" > "/dev/stderr"
            exit 2
        }
    }
    print "Instance", "Min", "Mean", "Std", "Runtime"
    next
}
{
    print $column["Instance"], \
          $column["BestObjective"], \
          $column["MeanBestObjective"], \
          $column["StdBestObjective"], \
          $column["MeanRuntime"]
}
' "$input_file" > "$output_file"

echo "Generated $output_file"
