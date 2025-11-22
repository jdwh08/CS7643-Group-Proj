#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(pwd)/data}"

mkdir -p "$DATA_ROOT/HandLabeled/S1Hand/"
mkdir -p "$DATA_ROOT/HandLabeled/LabelHand/"
mkdir -p "$DATA_ROOT/splits/flood_handlabeled"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S1Weak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/"

echo "=== Syncing HandLabeled/S1Hand (tiles) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/ \
  "$DATA_ROOT/HandLabeled/S1Hand/"

echo "=== Syncing HandLabeled/LabelHand (masks) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/ \
  "$DATA_ROOT/HandLabeled/LabelHand/"

echo "=== Syncing splits/flood_handlabeled (CSV files) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/splits/flood_handlabeled/ \
  "$DATA_ROOT/splits/flood_handlabeled/"

echo "=== Preparing partial S1Weak subset (50 random tiles) ==="
TMP_ALL=$(mktemp)
TMP_SEL=$(mktemp)

echo "Listing all S1Weak URIs..."
gsutil ls gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak > "$TMP_ALL"

echo "Selecting 50 random S1Weak tiles..."
if command -v shuf >/dev/null 2>&1; then
  shuf -n 50 "$TMP_ALL" > "$TMP_SEL"
else
  # Fallback to Python if shuf is not available 
  python3 - "$TMP_ALL" "$TMP_SEL" << 'PY'
import sys, random
all_path, sel_path = sys.argv[1], sys.argv[2]
with open(all_path) as f:
    lines = [l.strip() for l in f if l.strip()]
random.shuffle(lines)
lines = lines[:50]
with open(sel_path, "w") as f:
    for l in lines:
        f.write(l + "\n")
PY
fi

echo "Copying selected S1Weak tiles and matching Otsu masks..."
while read -r uri; do
  [ -z "$uri" ] && continue
  echo "  Image: $uri"
  # Copy image
  gsutil cp "$uri" "$DATA_ROOT/WeaklyLabeled/S1Weak/"

  # Derive matching label URI
  filename=$(basename "$uri")
  # replace S1Weak with S1OtsuLabelWeak
  label_file="${filename/S1Weak/S1OtsuLabelWeak}"
  label_uri="gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/$label_file"

echo "  Label: $label_uri"
gsutil cp "$label_uri" "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/" || \
    echo "  WARNING: missing label for $uri"
done < "$TMP_SEL"

rm -f "$TMP_ALL" "$TMP_SEL"

echo "=== Done (local). Structure under $DATA_ROOT: ==="
echo "  HandLabeled/S1Hand"
echo "  HandLabeled/LabelHand"
echo "  splits/flood_handlabeled"
echo "  WeaklyLabeled/S1Weak (50 random tiles)"
echo "  WeaklyLabeled/S1OtsuLabelWeak (matching masks)"