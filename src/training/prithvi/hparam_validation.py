"""Aggregate hyperparameters and metrics across all model runs.

This script processes all model runs in logs_hand and logs_weak directories,
extracting hyperparameters from hparams.yaml and metrics from metrics.csv,
then saves the aggregated results to a single CSV file.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

OUTPUT_PATH = project_root / "outputs" / "prithvi"
LOGS_HAND = OUTPUT_PATH / "logs_hand"
LOGS_WEAK = OUTPUT_PATH / "logs_weak"


def load_params(model_path: Path) -> dict[str, Any]:
    """Load hyperparameters from hparams.yaml.

    Args:
        model_path: Path to the version directory containing hparams.yaml

    Returns:
        Dictionary of hyperparameters
    """
    yaml_file = model_path / "hparams.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"hparams.yaml not found in {model_path}")

    params_all = yaml.safe_load(yaml_file.read_text())

    # Extract parameters of interest (similar to eval.py load_params)
    params: dict[str, Any] = {}

    # Model arguments
    if "model_args" in params_all:
        params.update(**params_all["model_args"])
        # Remove nested structures that don't serialize well
        params.pop("necks", None)
        params.pop("backbone_indices", None)
        # Handle backbone_bands - convert list to string for CSV
        if "backbone_bands" in params and isinstance(params["backbone_bands"], list):
            params["backbone_bands"] = ",".join(params["backbone_bands"])
        # Handle decoder_channels - convert list to string for CSV
        if "decoder_channels" in params and isinstance(
            params["decoder_channels"], list
        ):
            params["decoder_channels"] = ",".join(map(str, params["decoder_channels"]))
        # Handle loss_metric - convert dict to string for CSV
        if "loss_metric" in params and isinstance(params["loss_metric"], dict):
            params["loss_metric"] = ",".join(
                [f"{k}:{v}" for k, v in params["loss_metric"].items()]
            )

    # Training hyperparameters
    params["loss"] = params_all.get("loss")
    params["lr"] = params_all.get("lr")
    params["optimizer"] = params_all.get("optimizer")

    # Optimizer hyperparameters
    if "optimizer_hparams" in params_all:
        for key, value in params_all["optimizer_hparams"].items():
            params[f"optimizer_{key}"] = value

    # Scheduler hyperparameters
    params["scheduler"] = params_all.get("scheduler")
    if "scheduler_hparams" in params_all:
        for key, value in params_all["scheduler_hparams"].items():
            params[f"scheduler_{key}"] = value

    return params


def extract_metrics(metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Extract metrics from metrics DataFrame.

    Strategy:
    - Train/Val: Extract metrics from the epoch with best validation mIoU
      (since the model checkpoint is saved based on best val mIoU)
    - Test: Extract final values (test is only run at the end)
    - Per-class metrics: Extract from the same rows as the main metrics

    Args:
        metrics_df: DataFrame loaded from metrics.csv

    Returns:
        Dictionary of extracted metrics
    """
    metrics: dict[str, Any] = {}

    # Convert epoch and step to numeric, handling NaN
    metrics_df = metrics_df.copy()
    metrics_df["epoch"] = pd.to_numeric(metrics_df["epoch"], errors="coerce")
    metrics_df["step"] = pd.to_numeric(metrics_df["step"], errors="coerce")

    # Identify metric columns
    train_cols = [col for col in metrics_df.columns if col.startswith("train/")]
    val_cols = [col for col in metrics_df.columns if col.startswith("val/")]
    test_cols = [col for col in metrics_df.columns if col.startswith("test/")]

    # Extract test metrics (final values, since test runs at the end)
    test_rows = metrics_df[metrics_df[test_cols].notna().any(axis=1)]
    if len(test_rows) > 0:
        # Get the last row with test metrics
        test_row = test_rows.iloc[-1]
        for col in test_cols:
            value = test_row[col]
            if pd.notna(value):
                # Remove 'test/' prefix for column name
                metric_name = col.replace("test/", "test_")
                metrics[metric_name] = value

    # Find the epoch with best validation mIoU
    val_rows = metrics_df[metrics_df[val_cols].notna().any(axis=1)]
    best_val_epoch = None
    best_val_row = None

    if len(val_rows) > 0 and "val/mIoU" in val_cols:
        # Find row with best validation mIoU
        best_val_idx = val_rows["val/mIoU"].idxmax()
        best_val_row = val_rows.loc[best_val_idx]
        best_val_epoch = best_val_row["epoch"]

        # Extract all validation metrics from this epoch
        for col in val_cols:
            value = best_val_row[col]
            if pd.notna(value):
                # Remove 'val/' prefix for column name
                metric_name = col.replace("val/", "val_")
                metrics[metric_name] = value

    # Extract train metrics from the same epoch as best validation
    train_rows = metrics_df[metrics_df[train_cols].notna().any(axis=1)]
    if len(train_rows) > 0 and best_val_epoch is not None:
        # Find train metrics from the same epoch as best validation
        # Since train metrics are logged throughout the epoch, we'll take the last
        # train metrics logged during that epoch
        train_rows_in_epoch = train_rows[train_rows["epoch"] == best_val_epoch]

        if len(train_rows_in_epoch) > 0:
            # Take the last train metrics from this epoch (most recent)
            train_row = train_rows_in_epoch.iloc[-1]
        else:
            # Fallback: find the closest train metrics
            # (last train metrics before or at this epoch)
            train_rows_before = train_rows[train_rows["epoch"] <= best_val_epoch]
            if len(train_rows_before) > 0:
                train_row = train_rows_before.iloc[-1]
            else:
                # If no train metrics before this epoch, take the first available
                train_row = train_rows.iloc[0]

        # Extract all train metrics
        for col in train_cols:
            value = train_row[col]
            if pd.notna(value):
                # Remove 'train/' prefix for column name
                metric_name = col.replace("train/", "train_")
                metrics[metric_name] = value

    return metrics


def process_version_directory(
    version_path: Path, data_source: str, version_num: int
) -> dict[str, Any]:
    """Process a single version directory.

    Args:
        version_path: Path to the version directory
        data_source: Either "hand" or "weak"
        version_num: Version number (extracted from directory name)

    Returns:
        Dictionary containing all parameters and metrics for this run
    """
    result: dict[str, Any] = {}

    # Add metadata
    result["data_source"] = data_source
    result["version"] = version_num

    # Load hyperparameters
    try:
        params = load_params(version_path)
        result.update(params)
    except Exception as e:
        print(f"Warning: Failed to load params from {version_path}: {e}")

    # Load metrics
    metrics_file = version_path / "metrics.csv"
    if metrics_file.exists():
        try:
            metrics_df = pd.read_csv(metrics_file)
            metrics = extract_metrics(metrics_df)
            result.update(metrics)
        except Exception as e:
            print(f"Warning: Failed to load metrics from {metrics_file}: {e}")
    else:
        print(f"Warning: metrics.csv not found in {version_path}")

    return result


def aggregate_all_runs() -> pd.DataFrame:
    """Aggregate all runs from logs_hand and logs_weak.

    Returns:
        DataFrame with one row per model run, containing all parameters and metrics
    """
    all_runs: list[dict[str, Any]] = []

    # Process logs_hand
    if LOGS_HAND.exists():
        for version_dir in sorted(LOGS_HAND.iterdir()):
            if version_dir.is_dir() and version_dir.name.startswith("version_"):
                try:
                    version_num = int(version_dir.name.replace("version_", ""))
                    run_data = process_version_directory(
                        version_dir, "hand", version_num
                    )
                    all_runs.append(run_data)
                except ValueError:
                    print(
                        f"Warning: Could not parse version number from {version_dir.name}"
                    )
                except Exception as e:
                    print(f"Error processing {version_dir}: {e}")

    # Process logs_weak
    if LOGS_WEAK.exists():
        for version_dir in sorted(LOGS_WEAK.iterdir()):
            if version_dir.is_dir() and version_dir.name.startswith("version_"):
                try:
                    version_num = int(version_dir.name.replace("version_", ""))
                    run_data = process_version_directory(
                        version_dir, "weak", version_num
                    )
                    all_runs.append(run_data)
                except ValueError:
                    print(
                        f"Warning: Could not parse version number from {version_dir.name}"
                    )
                except Exception as e:
                    print(f"Error processing {version_dir}: {e}")

    # Convert to DataFrame
    if not all_runs:
        print("Warning: No runs found to process")
        return pd.DataFrame()

    df = pd.DataFrame(all_runs)

    # Reorder columns: metadata first, then params, then metrics
    metadata_cols = ["data_source", "version"]
    param_cols = [
        col
        for col in df.columns
        if col not in metadata_cols
        and not any(col.startswith(prefix) for prefix in ["train_", "val_", "test_"])
    ]
    metric_cols = [col for col in df.columns if col not in metadata_cols + param_cols]

    # Sort metric columns: test, then val, then train
    def metric_sort_key(col: str) -> tuple[int, str]:
        """Sort key for metric columns."""
        if col.startswith("test_"):
            return (0, col)
        elif col.startswith("val_"):
            return (1, col)
        elif col.startswith("train_"):
            return (2, col)
        else:
            return (3, col)

    metric_cols_sorted = sorted(metric_cols, key=metric_sort_key)

    df = df[metadata_cols + param_cols + metric_cols_sorted]

    return df


def main() -> None:
    """Main function to aggregate and save results."""
    print("Aggregating hyperparameters and metrics from all runs...")

    df = aggregate_all_runs()

    if df.empty:
        print("No data to save.")
        return

    # Save to CSV
    output_file = OUTPUT_PATH / "hparam_validation_results.csv"
    df.to_csv(output_file, index=False)

    print(f"\nSuccessfully processed {len(df)} runs")
    print(f"Results saved to: {output_file}")
    print(f"\nColumns ({len(df.columns)} total):")
    print(
        f"  Metadata: {len([c for c in df.columns if c in ['data_source', 'version']])}"
    )
    param_count = len(
        [
            c
            for c in df.columns
            if c not in ["data_source", "version"]
            and not any(c.startswith(p) for p in ["train_", "val_", "test_"])
        ]
    )
    print(f"  Parameters: {param_count}")
    metric_count = len(
        [
            c
            for c in df.columns
            if any(c.startswith(p) for p in ["train_", "val_", "test_"])
        ]
    )
    print(f"  Metrics: {metric_count}")
    print(f"\nFirst few columns: {list(df.columns[:10])}")


if __name__ == "__main__":
    main()
