#!/bin/bash
# Unzip the organizer dataset ONCE into data/ (the zip is never copied or committed).
#   bash fr3duo_pipeline/tools/unpack_dataset.sh [rand500|handover84|sample10]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIPDIR="$ROOT/C2_Teach-the-robot_VLA/Dataset"
case "${1:-rand500}" in
  rand500)    ZIP="$ZIPDIR/fr3_duo_rand500.zip";                 OUT="fr3_duo_rand500" ;;
  handover84) ZIP="$ZIPDIR/fr3_duo_handover_84_ep.zip";          OUT="fr3_duo_handover_v9_night_84" ;;
  sample10)   ZIP="$ZIPDIR/FR3Duo_Handover_Dataset_lerobot3.zip"; OUT="FR3Duo_Handover_Dataset_lerobot3" ;;
  *) echo "unknown dataset '$1'" >&2; exit 2 ;;
esac
[ -f "$ZIP" ] || { echo "zip not found: $ZIP (copy it from the organizer drop)" >&2; exit 2; }
mkdir -p "$ROOT/data"
if [ -f "$ROOT/data/$OUT/meta/info.json" ]; then
  echo "already unpacked: data/$OUT"; exit 0
fi
echo "unzipping $(basename "$ZIP") -> data/$OUT"
unzip -q -o "$ZIP" -d "$ROOT/data/"
( cd "$ZIPDIR" && md5sum "$(basename "$ZIP")" ) > "$ROOT/data/$(basename "$ZIP").md5"
echo "done: $(find "$ROOT/data/$OUT" -type f | wc -l) files; md5 of the zip in data/$(basename "$ZIP").md5"
