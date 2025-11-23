#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(pwd)/data}"

mkdir -p "$DATA_ROOT/HandLabeled/S1Hand/"
mkdir -p "$DATA_ROOT/HandLabeled/S2Hand/"
mkdir -p "$DATA_ROOT/HandLabeled/LabelHand/"
mkdir -p "$DATA_ROOT/splits/flood_handlabeled"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S1Weak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S2Weak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S2OtsuLabelWeak/"

echo "=== Syncing HandLabeled/S1Hand (tiles) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/ \
  "$DATA_ROOT/HandLabeled/S1Hand/"

echo "=== Syncing HandLabeled/S2Hand (tiles) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/ \
  "$DATA_ROOT/HandLabeled/S2Hand/"

echo "=== Syncing HandLabeled/LabelHand (masks) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/ \
  "$DATA_ROOT/HandLabeled/LabelHand/"

echo "=== Syncing splits/flood_handlabeled (CSV files) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/splits/flood_handlabeled/ \
  "$DATA_ROOT/splits/flood_handlabeled/"

echo "=== Preparing partial S1Weak subset (50 deterministic random tiles) ==="
TMP_ALL=$(mktemp)
TMP_SEL=$(mktemp)

# Use a deterministic seed file for reproducible selection
# This seed file should be committed to version control
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED_FILE="${SCRIPT_DIR}/.sync_seed"
SEED_VALUE="${SEED_VALUE:31415926}"

# Create seed file if it doesn't exist
if [ ! -f "$SEED_FILE" ]; then
  echo "$SEED_VALUE" > "$SEED_FILE"
  echo "Created seed file: $SEED_FILE (seed: $SEED_VALUE)"
fi

echo "Listing all S1Weak URIs..."
gsutil ls gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak > "$TMP_ALL"

echo "Selecting 50 deterministic random S1Weak tiles (using seed from $SEED_FILE)..."
SEED=$(cat "$SEED_FILE")
if command -v shuf >/dev/null 2>&1; then
  # Generate deterministic random bytes from seed for shuf --random-source
  # Generate enough bytes (8KB) to handle large file lists
  TMP_SEED_BYTES=$(mktemp)
  python3 -c "import random; random.seed($SEED); open('$TMP_SEED_BYTES', 'wb').write(bytes([random.randint(0, 255) for _ in range(8192)]))"
  shuf --random-source="$TMP_SEED_BYTES" -n 50 "$TMP_ALL" > "$TMP_SEL"
  rm -f "$TMP_SEED_BYTES"
else
  # Fallback to Python if shuf is not available
  python3 - "$TMP_ALL" "$TMP_SEL" "$SEED" << 'PY'
import sys, random
all_path, sel_path, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
random.seed(seed)
with open(all_path) as f:
    lines = [l.strip() for l in f if l.strip()]
random.shuffle(lines)
lines = lines[:50]
with open(sel_path, "w") as f:
    for l in lines:
        f.write(l + "\n")
PY
fi

echo "Copying selected S1Weak/S2Weak tiles and matching Otsu masks..."
while read -r uri; do
  [ -z "$uri" ] && continue
  filename=$(basename "$uri")
  
  # === S1 Downloads ===
  echo "  S1 Image: $uri"
  gsutil cp "$uri" "$DATA_ROOT/WeaklyLabeled/S1Weak/"

  # Derive matching S1 label URI
  s1_label_file="${filename/S1Weak/S1OtsuLabelWeak}"
  s1_label_uri="gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/$s1_label_file"
  echo "  S1 Label: $s1_label_uri"
  gsutil cp "$s1_label_uri" "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/" || \
    echo "  WARNING: missing S1 label for $uri"

  # === S2 Downloads ===
  # Derive matching S2 image filename (replace S1Weak with S2Weak)
  s2_filename="${filename/S1Weak/S2Weak}"
  s2_uri="gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2Weak/$s2_filename"
  echo "  S2 Image: $s2_uri"
  gsutil cp "$s2_uri" "$DATA_ROOT/WeaklyLabeled/S2Weak/" || \
    echo "  WARNING: missing S2 image for $uri"

  # Derive matching S2 label URI (replace S1Weak with S2OtsuLabelWeak)
  s2_label_file="${filename/S1Weak/S2IndexLabelWeak}"
  s2_label_uri="gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/$s2_label_file"
  echo "  S2 Label: $s2_label_uri"
  gsutil cp "$s2_label_uri" "$DATA_ROOT/WeaklyLabeled/S2IndexLabelWeak/" || \
    echo "  WARNING: missing S2 label for $uri"
done < "$TMP_SEL"

rm -f "$TMP_ALL" "$TMP_SEL"

echo "=== Done (local). Structure under $DATA_ROOT: ==="
echo "  HandLabeled/S1Hand"
echo "  HandLabeled/S2Hand"
echo "  HandLabeled/LabelHand"
echo "  splits/flood_handlabeled"
echo "  WeaklyLabeled/S1Weak (50 deterministic random tiles)"
echo "  WeaklyLabeled/S1OtsuLabelWeak (matching masks)"
echo "  WeaklyLabeled/S2Weak (matching tiles for same 50 samples)"
echo "  WeaklyLabeled/S2OtsuLabelWeak (matching masks)"