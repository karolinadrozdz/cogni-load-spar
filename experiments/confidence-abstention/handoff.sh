#!/usr/bin/env bash
# Wait for a Modal run to finish, fetch its results, and run the analysis locally.
#
#   ./handoff.sh <tag>                 # tag = --tag given to abstention_modal.py
#   ABSTENTION_RESULTS_VOLUME=... ./handoff.sh <tag>   # if the run used a custom volume
#
# Needs: `pip install -r requirements.txt` and a Modal token for the spar-f26 workspace
# (`modal token new`, then `modal profile activate spar-f26`). Pulls into results/<tag>/.
set -euo pipefail
tag="${1:?usage: ./handoff.sh <tag>}"
vol="${ABSTENTION_RESULTS_VOLUME:-abstention-results}"

# params.json is written last, so it marks a complete run. Poll every 5 minutes.
until modal volume ls "$vol" "$tag" 2>/dev/null | grep -q params.json; do
  echo "$(date '+%H:%M') $vol/$tag not finished (no params.json yet); waiting 5 min"
  sleep 300
done

mkdir -p results
rm -rf "results/$tag"
modal volume get "$vol" "$tag" results/
echo "fetched results/$tag:"; ls -la "results/$tag"
python analyze.py "results/$tag"
