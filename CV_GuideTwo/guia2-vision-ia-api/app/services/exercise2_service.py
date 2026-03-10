from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.contracts.exercise_contract import ExerciseContract
from app.schemas.execution import ExerciseRequest, ExerciseResponse, ExerciseType, ImageArtifact
from app.utils.image_utils import image_to_base64
from app.utils.static_url import build_static_url


class Exercise2Service(ExerciseContract):
    @property
    def exercise_type(self) -> str:
        return ExerciseType.ejercicio2.value

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        out_dir = Path("app/output/ejercicio2")
        out_dir.mkdir(parents=True, exist_ok=True)
        expected_models = ["NaiveBayes", "SVM"]

        # Cargar dataset Iris
        iris = load_iris()
        x, y = iris.data, iris.target
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(request.options.get("test_size", 0.3)),
            random_state=int(request.options.get("random_state", 42)),
            stratify=y,
        )

        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Diccionario de modelos con búsqueda de hiperparámetros
        models = {
            "NaiveBayes": GridSearchCV(
                GaussianNB(),
                param_grid={
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                },
                cv=cv,
            ),
            "SVM": GridSearchCV(
                # SVM es muy sensible a la escala de los datos, por lo que usamos un Pipeline
                Pipeline([("scaler", StandardScaler()), ("svm", SVC(random_state=42))]),
                param_grid={
                    "svm__C": [0.1, 1, 10, 100],
                    "svm__kernel": ["linear", "rbf", "poly"],
                    "svm__gamma": ["scale", "auto"]
                },
                cv=cv,
            ),
        }

        scores: dict[str, float] = {}
        reports: dict[str, dict] = {}
        best_parameters: dict[str, dict] = {}
        images: list[ImageArtifact] = []

        for model_name, model in models.items():
            # Entrenamiento y predicción
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            # Recolección de métricas e hiperparámetros
            scores[model_name] = float(accuracy_score(y_test, y_pred))
            best_parameters[model_name] = model.best_params_
            reports[model_name] = classification_report(
                y_test,
                y_pred,
                target_names=list(iris.target_names),
                output_dict=True,
            )

            # Generación y guardado de matriz de confusión
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
                display_labels=iris.target_names,
                cmap="Blues",
            )
            plt.title(f"Matriz de confusion - {model_name}")
            plt.tight_layout()

            img_name = f"confusion_matrix_{model_name}.png".replace(" ", "_")
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

        # Validación del contrato
        image_names = {image.name for image in images}
        missing_images = [
            f"confusion_matrix_{model_name}.png".replace(" ", "_")
            for model_name in expected_models
            if f"confusion_matrix_{model_name}.png".replace(" ", "_") not in image_names
        ]

        if missing_images:
            raise RuntimeError(
                f"Ejercicio2 incompleto: faltan imagenes esperadas: {', '.join(missing_images)}"
            )

        best_model = max(scores.items(), key=lambda item: item[1])

        # Retornar respuesta estructurada para el JSON
        return ExerciseResponse(
            type=ExerciseType.ejercicio2,
            summary={
                "best_model": best_model[0],
                "best_accuracy": best_model[1],
                "scores": scores,
                "best_params": best_parameters,
                "reports": reports,
            },
            images=images,
        )