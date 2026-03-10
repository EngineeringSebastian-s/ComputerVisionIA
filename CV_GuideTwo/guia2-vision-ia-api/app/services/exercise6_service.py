from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from app.contracts.exercise_contract import ExerciseContract
from app.schemas.execution import ExerciseRequest, ExerciseResponse, ExerciseType, ImageArtifact
from app.utils.image_utils import image_to_base64
from app.utils.static_url import build_static_url


class Exercise6Service(ExerciseContract):
    @property
    def exercise_type(self) -> str:
        # Asegúrate de tener 'ejercicio6' definido en tu Enum ExerciseType
        return ExerciseType.ejercicio6.value

    def _load_cats_dogs(
            self,
            root_dir: Path,
            resize: tuple[int, int] = (64, 64),
            max_samples_per_class: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []

        # Asumimos que dentro de root_dir hay subcarpetas como "Cats" y "Dogs"
        classes = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])

        for cls in classes:
            img_paths = list((root_dir / cls).rglob("*"))
            random.seed(42)
            random.shuffle(img_paths)  # Mezclar para tomar una muestra representativa

            loaded_count = 0
            for img_path in img_paths:
                if loaded_count >= max_samples_per_class:
                    break
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                # Cargar en escala de grises para reducir dimensionalidad y tiempo de cómputo en SVM
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Redimensionar y aplanar (flatten)
                img_resized = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                img_flattened = img_resized.flatten().astype(np.float32)

                x.append(img_flattened)
                y.append(cls)
                loaded_count += 1

        if not x:
            raise RuntimeError(f"No se cargaron imagenes en la ruta {root_dir}")
        return np.array(x, dtype=np.float32), np.array(y, dtype=object)

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        # Ajusta la ruta según la estructura real de tu proyecto
        project_root = Path(__file__).resolve().parents[2]
        dataset_dir = project_root / "dataset" / "Cats&Dogs"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"No existe el dataset en {dataset_dir}")

        out_dir = Path("app/output/ejercicio6")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Cargar datos (Redimensionados a 64x64 y limitados para no saturar el servidor)
        x, y = self._load_cats_dogs(dataset_dir, resize=(64, 64), max_samples_per_class=600)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = list(le.classes_)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y_enc, test_size=0.3, random_state=42, stratify=y_enc
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # Definir modelos a comparar según la guía:
        # 1. SVM por defecto (Baseline)
        # 2. SVM con hiperparámetros modificados (Optimizado)
        # 3. Otro tipo de modelo (Random Forest)
        models = {
            "SVM_Baseline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(random_state=42))  # Parámetros por defecto
            ]),
            "SVM_Optimized": GridSearchCV(
                Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=42))]),
                param_grid={
                    "clf__C": [0.1, 1, 10],
                    "clf__kernel": ["rbf", "linear"]
                },
                cv=cv,
                n_jobs=-1
            ),
            "RandomForest_Alternative": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        }

        scores: dict[str, float] = {}
        reports: dict[str, dict] = {}
        images: list[ImageArtifact] = []
        rows = []

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            acc = float(accuracy_score(y_test, y_pred))
            f1m = float(f1_score(y_test, y_pred, average="macro"))

            scores[model_name] = acc
            reports[model_name] = classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True
            )

            # Generar gráfica de matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap="Blues")
            plt.title(f"Matriz de Confusión - {model_name}")
            plt.tight_layout()

            img_name = f"confusion_matrix_{model_name}.png"
            img_path = out_dir / img_name
            plt.savefig(img_path, dpi=200)
            plt.close()

            images.append(
                ImageArtifact(
                    name=img_name,
                    path=str(img_path),
                    url=build_static_url(img_path),
                    content_base64=image_to_base64(img_path),
                )
            )

            rows.append({
                "model": model_name,
                "accuracy": acc,
                "f1_macro": f1m
            })

        best_model = max(scores.items(), key=lambda item: item[1])

        return ExerciseResponse(
            type=ExerciseType.ejercicio6,
            summary={
                "best_model": best_model[0],
                "best_accuracy": best_model[1],
                "comparison_table": rows,
                "reports": reports,
            },
            images=images,
        )