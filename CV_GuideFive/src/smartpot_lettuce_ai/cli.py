"""Command line interface for the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import ProjectConfig, TrainingConfig
from .dataset import load_dataset_metadata, to_folder_class_names
from .evaluate import evaluate_model
from .predict import predict_image
from .train import train_from_directory


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""

    parser = argparse.ArgumentParser(
        prog="smartpot-lettuce-ai",
        description="Herramientas para el proyecto de visión por computadora de lechuga.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Raíz del proyecto. Por defecto se autodetecta a partir de pyproject.toml.",
    )

    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser(
        "inspect-metadata",
        help="Mostrar un resumen del metadata del dataset de Kaggle.",
    )
    inspect_parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Ruta al archivo lettuce-diseases-metadata.json.",
    )
    inspect_parser.set_defaults(func=_cmd_inspect_metadata)

    train_parser = subparsers.add_parser(
        "train",
        help="Entrenar MobileNetV2 con el dataset organizado por carpetas de clase.",
    )
    train_parser.add_argument("--data-dir", type=Path, default=None, help="Directorio con imágenes.")
    train_parser.add_argument("--epochs", type=int, default=None, help="Número de épocas.")
    train_parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Número de épocas para fine-tuning.",
    )
    train_parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Porcentaje de validación.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño del batch.",
    )
    train_parser.add_argument("--seed", type=int, default=None, help="Semilla aleatoria.")
    train_parser.set_defaults(func=_cmd_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluar un modelo exportado sobre un directorio de prueba.",
    )
    evaluate_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Ruta al archivo .keras del modelo.",
    )
    evaluate_parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directorio de prueba organizado por clases.",
    )
    evaluate_parser.add_argument(
        "--class-names-path",
        type=Path,
        default=None,
        help="Archivo JSON con el orden de las clases.",
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predecir la clase de una sola imagen.",
    )
    predict_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Ruta al archivo .keras del modelo.",
    )
    predict_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Imagen a clasificar.",
    )
    predict_parser.add_argument(
        "--class-names-path",
        type=Path,
        default=None,
        help="Archivo JSON con el orden de las clases.",
    )
    predict_parser.set_defaults(func=_cmd_predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "command", None) is None:
        return _cmd_inspect_metadata(argparse.Namespace(root=args.root, metadata=None))

    return int(args.func(args))


def _cmd_inspect_metadata(args: argparse.Namespace) -> int:
    project = ProjectConfig.from_root(args.root, ensure_directories=False)
    metadata_path = args.metadata or project.metadata_path
    metadata = load_dataset_metadata(metadata_path)

    summary = {
        "name": metadata.name,
        "source_url": metadata.source_url,
        "class_names": list(metadata.class_names),
        "folder_class_names": list(to_folder_class_names(metadata.class_names)),
        "num_classes": len(metadata.class_names),
        "distribution": metadata.distribution,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _build_project_and_training(args: argparse.Namespace) -> tuple[ProjectConfig, TrainingConfig]:
    project = ProjectConfig.from_root(args.root, ensure_directories=False)
    if args.batch_size is not None:
        project.batch_size = args.batch_size
    if args.seed is not None:
        project.seed = args.seed

    training = TrainingConfig(
        epochs=args.epochs if args.epochs is not None else project.training.epochs,
        fine_tune_epochs=(
            args.fine_tune_epochs
            if args.fine_tune_epochs is not None
            else project.training.fine_tune_epochs
        ),
        validation_split=(
            args.validation_split
            if args.validation_split is not None
            else project.training.validation_split
        ),
    )
    project.training = training
    return project, training


def _cmd_train(args: argparse.Namespace) -> int:
    project, training = _build_project_and_training(args)
    data_dir = args.data_dir or project.paths.data_raw

    artifacts = train_from_directory(
        data_dir,
        project,
        training,
        class_names=project.class_names,
    )
    print(
        json.dumps(
            {
                "model_path": str(artifacts.model_path),
                "history_path": str(artifacts.history_path),
                "class_names_path": str(artifacts.class_names_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    project = ProjectConfig.from_root(args.root, ensure_directories=False)
    class_names = None
    if args.class_names_path is not None:
        class_names = json.loads(Path(args.class_names_path).read_text(encoding="utf-8"))

    artifacts = evaluate_model(
        args.model_path,
        args.test_dir,
        project,
        class_names=class_names,
    )
    print(
        json.dumps(
            {
                "metrics_path": str(artifacts.metrics_path),
                "report_path": str(artifacts.report_path),
                "confusion_matrix_path": str(artifacts.confusion_matrix_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    project = ProjectConfig.from_root(args.root, ensure_directories=False)
    class_names = None
    if args.class_names_path is not None:
        class_names = json.loads(Path(args.class_names_path).read_text(encoding="utf-8"))

    result = predict_image(
        args.model_path,
        args.image,
        project,
        class_names=class_names,
        class_names_path=args.class_names_path,
    )
    print(
        json.dumps(
            {
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "probabilities": list(result.probabilities),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0
