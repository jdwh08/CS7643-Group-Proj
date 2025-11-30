#!/usr/bin/env bash
set -euo pipefail

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

echo "=== Syncing WeaklyLabeled/S1Weak (full) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/ \
  "$DATA_ROOT/WeaklyLabeled/S1Weak/"

echo "=== Syncing WeaklyLabeled/S1OtsuLabelWeak (full) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/ \
  "$DATA_ROOT/WeaklyLabeled/S1OtsuLabelWeak/"

echo "=== Syncing WeaklyLabeled/S2Weak (full) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2Weak/ \
  "$DATA_ROOT/WeaklyLabeled/S2Weak/"

echo "=== Syncing WeaklyLabeled/S2IndexLabelWeak (full) ==="
gsutil -m rsync -r \
  gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak/ \
  "$DATA_ROOT/WeaklyLabeled/S2IndexLabelWeak/"

echo "=== Done (full bucket). Structure under $DATA_ROOT: ==="
echo "  HandLabeled/S1Hand"
echo "  HandLabeled/S2Hand"
echo "  HandLabeled/LabelHand"
echo "  splits/flood_handlabeled"
echo "  WeaklyLabeled/S1Weak (full)"
echo "  WeaklyLabeled/S1OtsuLabelWeak (full)"
echo "  WeaklyLabeled/S2Weak (full)"
echo "  WeaklyLabeled/S2IndexLabelWeak (full)"