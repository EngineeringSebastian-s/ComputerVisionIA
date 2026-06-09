"""Dataset helpers for the lettuce disease classification workflow."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import DEFAULT_LETTUCE_CLASS_NAMES
from .utils import folderify_label, load_json

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")
CLASS_NAME_PATTERN = re.compile(r"^\s*\d+\.\s*(?P<label>[^:\n]+?)\s*:", re.MULTILINE)


@dataclass(slots=True, frozen=True)
class ImageSample:
    """A single image path and its class label."""

    path: Path
    label: str


@dataclass(slots=True)
class DatasetMetadata:
    """Parsed metadata from the Kaggle dataset card."""

    name: str
    description: str
    class_names: tuple[str, ...]
    source_url: str | None = None
    distribution: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DatasetSplit:
    """Stratified train/validation/test split."""

    train: tuple[ImageSample, ...]
    validation: tuple[ImageSample, ...]
    test: tuple[ImageSample, ...]

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def extract_class_names_from_description(description: str) -> tuple[str, ...]:
    """Extract the class names enumerated in the dataset description."""

    matches = [match.strip() for match in CLASS_NAME_PATTERN.findall(description)]
    if matches:
        # Preserve order and remove accidental duplicates.
        return tuple(dict.fromkeys(matches))
    return DEFAULT_LETTUCE_CLASS_NAMES


def to_folder_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    """Convert human-readable class names to the folder naming convention used on disk."""

    return tuple(folderify_label(name) for name in class_names)


def load_dataset_metadata(metadata_path: Path | str) -> DatasetMetadata:
    """Load and normalize the dataset metadata file."""

    path = Path(metadata_path)
    raw = load_json(path)
    description = str(raw.get("description", ""))
    return DatasetMetadata(
        name=str(raw.get("name", path.stem)),
        description=description,
        class_names=extract_class_names_from_description(description),
        source_url=str(raw.get("url")) if raw.get("url") else None,
        distribution=list(raw.get("distribution", [])),
        raw=dict(raw),
    )


def collect_image_samples(
    data_dir: Path | str,
    *,
    class_names: Sequence[str] | None = None,
    allowed_extensions: Sequence[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> list[ImageSample]:
    """Collect image files from a directory tree organized by class name."""

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    labels = list(class_names) if class_names is not None else sorted(
        entry.name for entry in root.iterdir() if entry.is_dir()
    )
    extensions = {extension.lower() for extension in allowed_extensions}
    samples: list[ImageSample] = []

    for label in labels:
        class_dir = root / folderify_label(label)
        if not class_dir.exists():
            continue

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in extensions:
                samples.append(ImageSample(path=image_path, label=label))

    return samples


def count_samples_by_class(samples: Sequence[ImageSample]) -> dict[str, int]:
    """Count how many images belong to each class."""

    return dict(sorted(Counter(sample.label for sample in samples).items()))


def split_samples(
    samples: Sequence[ImageSample],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Split samples into stratified train/validation/test subsets."""

    if not math.isclose(train_ratio + validation_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("The split ratios must add up to 1.0.")

    groups: dict[str, list[ImageSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.label].append(sample)

    rng = random.Random(seed)
    train: list[ImageSample] = []
    validation: list[ImageSample] = []
    test: list[ImageSample] = []

    for label in sorted(groups):
        group = groups[label]
        rng.shuffle(group)
        total = len(group)

        train_count = int(round(total * train_ratio))
        validation_count = int(round(total * validation_ratio))
        if train_count + validation_count > total:
            overflow = train_count + validation_count - total
            validation_count = max(0, validation_count - overflow)
            overflow = max(0, train_count + validation_count - total)
            if overflow:
                train_count = max(0, train_count - overflow)

        split_point = train_count + validation_count
        train.extend(group[:train_count])
        validation.extend(group[train_count:split_point])
        test.extend(group[split_point:])

    return DatasetSplit(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
    )


def build_class_index_map(class_names: Sequence[str]) -> dict[str, int]:
    """Map class names to their numeric index."""

    return {label: index for index, label in enumerate(class_names)}
