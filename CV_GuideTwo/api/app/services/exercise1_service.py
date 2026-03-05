from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.contracts.exercise_contract import ExerciseContract
from app.schemas.execution import ExerciseRequest, ExerciseResponse, ExerciseType, ImageArtifact
from app.utils.image_utils import image_to_base64
from app.utils.static_url import build_static_url


class Exercise1Service(ExerciseContract):
    @property
    def exercise_type(self) -> str:
        return ExerciseType.ejercicio1.value

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        out_dir = Path("app/output/ejercicio1")
        out_dir.mkdir(parents=True, exist_ok=True)
        expected_models = ["LogisticRegression", "DecisionTree", "RandomForest"]

        iris = load_iris()
        x, y = iris.data, iris.target
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(request.options.get("test_size", 0.3)),
            random_state=int(request.options.get("random_state", 42)),
            stratify=y,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        models = {
            "LogisticRegression": Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
            ),
            "DecisionTree": GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                param_grid={
                    "max_depth": [2, 3, 4, 5, None],
                    "criterion": ["gini", "entropy"],
                    "min_samples_split": [2, 4, 6],
                },
                cv=cv,
            ),
            "RandomForest": GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid={
                    "n_estimators": [50, 100, 200],
                    "max_depth": [2, 3, 4, 5, None],
                    "max_features": ["sqrt", "log2", None],
                },
                cv=cv,
            ),
        }

        scores: dict[str, float] = {}
        reports: dict[str, dict] = {}
        images: list[ImageArtifact] = []

        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            scores[model_name] = float(accuracy_score(y_test, y_pred))
            reports[model_name] = classification_report(
                y_test,
                y_pred,
                target_names=list(iris.target_names),
                output_dict=True,
            )

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

        # Contrato: ejercicio 1 siempre debe exponer 3 matrices de confusion:
        # LogisticRegression, DecisionTree y RandomForest.
        image_names = {image.name for image in images}
        missing_images = [
            f"confusion_matrix_{model_name}.png".replace(" ", "_")
            for model_name in expected_models
            if f"confusion_matrix_{model_name}.png".replace(" ", "_") not in image_names
        ]
        if missing_images:
            raise RuntimeError(
                f"Ejercicio1 incompleto: faltan imagenes esperadas: {', '.join(missing_images)}"
            )

        best_model = max(scores.items(), key=lambda item: item[1])

        return ExerciseResponse(
            type=ExerciseType.ejercicio1,
            summary={
                "best_model": best_model[0],
                "best_accuracy": best_model[1],
                "scores": scores,
                "reports": reports,
            },
            images=images,
        )
