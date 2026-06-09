from __future__ import annotations

from pathlib import Path

from smartpot_lettuce_ai.config import ProjectConfig, TrainingConfig


def test_project_config_builds_expected_paths() -> None:
    root = Path("/tmp/smartpot-lettuce-ai")
    project = ProjectConfig.from_root(
        root,
        ensure_directories=False,
        training=TrainingConfig(epochs=3, fine_tune_epochs=1),
    )

    assert project.paths.root == root.resolve()
    assert project.paths.data_raw == root.resolve() / "data" / "raw"
    assert project.paths.models_exported == root.resolve() / "models" / "exported"
    assert project.training.epochs == 3
    assert project.training.fine_tune_epochs == 1
