from __future__ import annotations

from pathlib import Path

from smartpot_lettuce_ai.dataset import (
    ImageSample,
    collect_image_samples,
    extract_class_names_from_description,
    load_dataset_metadata,
    split_samples,
)


def test_extract_class_names_from_real_metadata() -> None:
    metadata_path = Path(__file__).resolve().parents[1] / "lettuce-diseases-metadata.json"
    metadata = load_dataset_metadata(metadata_path)

    assert metadata.name == "Lettuce Diseases"
    assert len(metadata.class_names) == 8
    assert metadata.class_names[0] == "Healthy"
    assert "Powdery mildew on lettuce" in metadata.class_names


def test_extract_class_names_from_description_falls_back_to_defaults() -> None:
    class_names = extract_class_names_from_description("This description does not list classes.")

    assert len(class_names) == 8
    assert class_names[0] == "Healthy"


def test_split_samples_preserves_labels_across_subsets() -> None:
    samples = [
        ImageSample(Path(f"/tmp/healthy_{index}.jpg"), "Healthy")
        for index in range(6)
    ] + [
        ImageSample(Path(f"/tmp/bacterial_{index}.jpg"), "Bacterial")
        for index in range(6)
    ]

    split = split_samples(samples, train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25, seed=7)

    assert len(split.train) + len(split.validation) + len(split.test) == len(samples)
    assert {sample.label for sample in split.train} == {"Healthy", "Bacterial"}
    assert {sample.label for sample in split.validation} == {"Healthy", "Bacterial"}
    assert {sample.label for sample in split.test} == {"Healthy", "Bacterial"}


def test_collect_image_samples_ignores_non_image_files(tmp_path: Path) -> None:
    healthy_dir = tmp_path / "Healthy"
    healthy_dir.mkdir()
    (healthy_dir / "leaf.jpg").write_bytes(b"fake")
    (healthy_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    samples = collect_image_samples(tmp_path, class_names=["Healthy"])

    assert len(samples) == 1
    assert samples[0].path.name == "leaf.jpg"
