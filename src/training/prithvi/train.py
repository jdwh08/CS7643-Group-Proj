"""Router to run Prithvi trainers based on YAML config."""

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from src.training.prithvi.train_hand import (
    PrithviHandConfig,
    PrithviHandTrainer,
)
from src.training.prithvi.train_weak import (
    PrithviWeakConfig,
    PrithviWeakTrainer,
)

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _deep_merge_dict(
    base: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Recursively merge override keys into a copy of base."""
    result = deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


class PrithviTrainer:
    """Load configuration and dispatch to the correct trainer."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.raw_config: dict[str, Any] = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r") as f:
            return yaml.safe_load(f) or {}

    def _resolve_overrides(self, data_source: str) -> dict[str, Any]:
        # Global overrides under "settings" are applied first, then dataset-specific block.
        base_overrides = self.raw_config.get("settings", {}) or {}
        dataset_overrides = self.raw_config.get(data_source, {}) or {}
        return _deep_merge_dict(base_overrides, dataset_overrides)

    def run(self) -> Path:
        data_source: Literal["weak", "hand"] = self.raw_config.get("data", "weak")
        overrides = self._resolve_overrides(data_source)

        if data_source == "weak":
            trainer_config = PrithviWeakConfig.from_overrides(overrides)
            trainer = PrithviWeakTrainer(trainer_config, self.config_path)
        elif data_source == "hand":
            trainer_config = PrithviHandConfig.from_overrides(overrides)
            trainer = PrithviHandTrainer(trainer_config, self.config_path)
        else:
            message = f"Unsupported data source '{data_source}'. Use 'weak' or 'hand'."
            raise ValueError(message)

        trainer.run()
        # Copy the config file to the output path
        shutil.copy(self.config_path, trainer.config.output_path / "config.yaml")
        return trainer.config.output_path


def main(config_path: Path | str | None = None) -> Path:
    trainer = PrithviTrainer(config_path=config_path)
    return trainer.run()


if __name__ == "__main__":
    main()
