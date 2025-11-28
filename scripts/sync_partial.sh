#!/usr/bin/env bash
set -euo pipefail

# Auto-detect PACE
if command -v pace-quota >/dev/null 2>&1; then
  IS_PACE=1
else
  IS_PACE=0
fi


DATA_ROOT="${DATA_ROOT:-$(pwd)/data}"
echo "[env] Using DATA_ROOT=$DATA_ROOT"

mkdir -p "$DATA_ROOT/HandLabeled/S1Hand/"
mkdir -p "$DATA_ROOT/HandLabeled/S2Hand/"
mkdir -p "$DATA_ROOT/HandLabeled/LabelHand/"
mkdir -p "$DATA_ROOT/splits/flood_handlabeled"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S1Weak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S2Weak/"
mkdir -p "$DATA_ROOT/WeaklyLabeled/S2IndexLabelWeak/"

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
SEED_VALUE="${SEED_VALUE:-31415926}"

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

echo "Copying selected S1Weak/S2Weak tiles and matching masks..."
# Build lists of URIs for batch copying (one list per destination directory)
TMP_S1_URIS=$(mktemp)
TMP_S1_LABEL_URIS=$(mktemp)
TMP_S2_URIS=$(mktemp)
TMP_S2_LABEL_URIS=$(mktemp)

# Build URI lists by processing all selected S1Weak URIs
while read -r uri; do
  [ -z "$uri" ] && continue
  filename=$(basename "$uri")
  
  # S1 image (original URI)
  echo "$uri" >> "$TMP_S1_URIS"
  
  # S1 label URI
  s1_label_file="${filename/S1Weak/S1OtsuLabelWeak}"
  echo "gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/$s1_label_file" >> "$TMP_S1_LABEL_URIS"
  
  # S2 image URI
  s2_filename="${filename/S1Weak/S2Weak}"
  echo "gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2Weak/$s2_filename" >> "$TMP_S2_URIS"
  
  # S2 label URI
  s2_label_file="${filename/S1Weak/S2IndexLabelWeak}"
  echo "gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/$s2_label_file" >> "$TMP_S2_LABEL_URIS"
done < "$TMP_SEL"

if [[ "$IS_PACE" -eq 1 ]]; then
# Batch copy all files in parallel using gsutil -m cp
# Use -n (no-clobber) to avoid re-downloading existing files
# Read URIs into arrays and expand them as arguments to gsutil
echo "  Copying S1 images..."
  if [ -s "$TMP_S1_URIS" ]; then
    mapfile -t s1_uris < "$TMP_S1_URIS"
    [ ${#s1_uris[@]} -gt 0 ] && gsutil -m cp -n "${s1_uris[@]}" "$DATA_ROOT/WeaklyLabeled/S1Weak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi

  echo "  Copying S1 labels..."
  if [ -s "$TMP_S1_LABEL_URIS" ]; then
    mapfile -t s1_label_uris < "$TMP_S1_LABEL_URIS"
    [ ${#s1_label_uris[@]} -gt 0 ] && gsutil -m cp -n "${s1_label_uris[@]}" "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi

  echo "  Copying S2 images..."
  if [ -s "$TMP_S2_URIS" ]; then
    mapfile -t s2_uris < "$TMP_S2_URIS"
    [ ${#s2_uris[@]} -gt 0 ] && gsutil -m cp -n "${s2_uris[@]}" "$DATA_ROOT/WeaklyLabeled/S2Weak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi

  echo "  Copying S2 labels..."
  if [ -s "$TMP_S2_LABEL_URIS" ]; then
    mapfile -t s2_label_uris < "$TMP_S2_LABEL_URIS"
    [ ${#s2_label_uris[@]} -gt 0 ] && gsutil -m cp -n "${s2_label_uris[@]}" "$DATA_ROOT/WeaklyLabeled/S2IndexLabelWeak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi
else
# gsutil -m cp and mapfile breaks on macOS, remove them and use plain gsutil cp
  echo "  Copying S1 images..."
  if [ -s "$TMP_S1_URIS" ]; then
    gsutil cp -n $(cat "$TMP_S1_URIS") "$DATA_ROOT/WeaklyLabeled/S1Weak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi

  echo "  Copying S1 labels..."
  if [ -s "$TMP_S1_LABEL_URIS" ]; then
    gsutil cp -n $(cat "$TMP_S1_LABEL_URIS") "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi

  echo "  Copying S2 images..."
  if [ -s "$TMP_S2_URIS" ]; then
    gsutil cp -n $(cat "$TMP_S2_URIS") "$DATA_ROOT/WeaklyLabeled/S2Weak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi

  echo "  Copying S2 labels..."
  if [ -s "$TMP_S2_LABEL_URIS" ]; then
    gsutil cp -n $(cat "$TMP_S2_LABEL_URIS") "$DATA_ROOT/WeaklyLabeled/S2IndexLabelWeak/" 2>&1 | grep -v "Skipping\|Copying" || true
  fi
fi

rm -f "$TMP_S1_URIS" "$TMP_S1_LABEL_URIS" "$TMP_S2_URIS" "$TMP_S2_LABEL_URIS"
rm -f "$TMP_ALL" "$TMP_SEL"

echo "=== Done (partial). Structure under $DATA_ROOT: ==="
echo "  HandLabeled/S1Hand"
echo "  HandLabeled/S2Hand"
echo "  HandLabeled/LabelHand"
echo "  splits/flood_handlabeled"
echo "  WeaklyLabeled/S1Weak (50 deterministic random tiles)"
echo "  WeaklyLabeled/S1OtsuLabelWeak (matching masks)"
echo "  WeaklyLabeled/S2Weak (matching tiles for same 50 samples)"
echo "  WeaklyLabeled/S2IndexLabelWeak (matching masks)"