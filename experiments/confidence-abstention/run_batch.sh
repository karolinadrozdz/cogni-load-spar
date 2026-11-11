#!/usr/bin/env bash
# Launch one batch of the Olmo-32B sweep concurrently: ./run_batch.sh 4   (batches in sweep_batches.tsv; 12 = in-trace probes)
# `modal run --detach` blocks until its run ends, so each is backgrounded with nohup; runs survive the terminal closing.
# Per-run client output: logs/<tag>.log
set -euo pipefail
cd "$(dirname "$0")"
[ $# -eq 1 ] || { echo "usage: $0 <batch number>"; exit 1; }
mkdir -p logs
awk -F"\t" -v n="$1" '$1 == n {print $2}' sweep_batches.tsv | while read -r cmd; do
    tag=$(sed -E 's/.*--tag ([^ ]+).*/\1/' <<< "$cmd")
    nohup bash -c "$cmd" </dev/null > "logs/$tag.log" 2>&1 &
    echo "launched $tag (log: logs/$tag.log)"
    sleep 2
done
