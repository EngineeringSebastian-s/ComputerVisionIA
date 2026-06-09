"""Configuration objects for the SmartPot lettuce AI project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .utils import ensure_directory, find_project_root

DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 20
DEFAULT_FINE_TUNE_EPOCHS = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_FINE_TUNE_LEARNING_RATE = 1e-5
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_FINE_TUNE_AT = 100

DEFAULT_LETTUCE_CLASS_NAMES: tuple[str, ...] = (
    "Healthy",
    "Bacterial",
    "Downy mildew on lettuce",
    "Powdery mildew on lettuce",
    "Septoria blight on lettuce",
    "Shepherd purse weeds",
    "Viral",
    "Wilt and leaf blight on lettuce",
)


@dataclass(slots=True)
class ProjectPaths:
    """Canonical filesystem locations used by the project."""

    root: Path
    data_raw: Path
    data_processed: Path
    data_splits: Path
    models_checkpoints: Path
    models_exported: Path
    reports_figures: Path
    reports_metrics: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        return cls(
            root=root,
            data_raw=root / "data" / "raw",
            data_processed=root / "data" / "processed",
            data_splits=root / "data" / "splits",
            models_checkpoints=root / "models" / "checkpoints",
            models_exported=root / "models" / "exported",
            reports_figures=root / "reports" / "figures",
            reports_metrics=root / "reports" / "metrics",
        )

    def ensure(self) -> "ProjectPaths":
        """Create every configured directory."""

        for path in (
            self.data_raw,
            self.data_processed,
            self.data_splits,
            self.models_checkpoints,
            self.models_exported,
            self.reports_figures,
            self.reports_metrics,
        ):
            ensure_directory(path)
        return self

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "data_raw": str(self.data_raw),
            "data_processed": str(self.data_processed),
            "data_splits": str(self.data_splits),
            "models_checkpoints": str(self.models_checkpoints),
            "models_exported": str(self.models_exported),
            "reports_figures": str(self.reports_figures),
            "reports_metrics": str(self.reports_metrics),
        }


@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters for the MobileNetV2 workflow."""

    epochs: int = DEFAULT_EPOCHS
    fine_tune_epochs: int = DEFAULT_FINE_TUNE_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    fine_tune_learning_rate: float = DEFAULT_FINE_TUNE_LEARNING_RATE
    dropout_rate: float = DEFAULT_DROPOUT_RATE
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    test_split: float = DEFAULT_TEST_SPLIT
    fine_tune_at: int = DEFAULT_FINE_TUNE_AT
    early_stopping_patience: int = 5

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class ProjectConfig:
    """Top-level configuration for the project."""

    paths: ProjectPaths
    metadata_path: Path
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    seed: int = DEFAULT_SEED
    class_names: tuple[str, ...] = DEFAULT_LETTUCE_CLASS_NAMES
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_root(
        cls,
        root: Path | None = None,
        *,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = DEFAULT_SEED,
        class_names: Sequence[str] = DEFAULT_LETTUCE_CLASS_NAMES,
        training: TrainingConfig | None = None,
        ensure_directories: bool = True,
    ) -> "ProjectConfig":
        """Build a config from a repository root."""

        project_root = find_project_root(root or Path(__file__))
        paths = ProjectPaths.from_root(project_root)
        if ensure_directories:
            paths.ensure()

        return cls(
            paths=paths,
            metadata_path=project_root / "lettuce-diseases-metadata.json",
            image_size=image_size,
            batch_size=batch_size,
            seed=seed,
            class_names=tuple(class_names),
            training=training or TrainingConfig(),
        )

    def ensure_directories(self) -> "ProjectConfig":
        """Create the filesystem layout used by the project."""

        self.paths.ensure()
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "paths": self.paths.to_dict(),
            "metadata_path": str(self.metadata_path),
            "image_size": list(self.image_size),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "class_names": list(self.class_names),
            "training": self.training.to_dict(),
        }
