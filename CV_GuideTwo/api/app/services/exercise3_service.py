from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.contracts.exercise_contract import ExerciseContract
from app.schemas.execution import ExerciseRequest, ExerciseResponse, ExerciseType, ImageArtifact
from app.utils.image_utils import image_to_base64
from app.utils.static_url import build_static_url


class Exercise3Service(ExerciseContract):
    @property
    def exercise_type(self) -> str:
        return ExerciseType.ejercicio3.value

    @staticmethod
    def _map_clusters_to_labels(y_true: np.ndarray, clusters: np.ndarray) -> dict[int, int]:
        cm = confusion_matrix(y_true, clusters)
        cost = cm.max() - cm
        row_ind, col_ind = linear_sum_assignment(cost)
        return {cluster: label for label, cluster in zip(row_ind, col_ind)}

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        out_dir = Path("app/output/ejercicio3")
        out_dir.mkdir(parents=True, exist_ok=True)

        iris = load_iris()
        x, y = iris.data, iris.target
        target_names = list(iris.target_names)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=float(request.options.get("test_size", 0.3)),
            stratify=y,
            random_state=int(request.options.get("random_state", 42)),
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        k_values = list(range(1, int(request.options.get("k_max", 10)) + 1))
        inertias = []
        silhouettes = []

        for k in k_values:
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=20,
                max_iter=500,
                random_state=42,
            )
            km.fit(x_train_scaled)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(x_train_scaled, km.labels_)) if k >= 2 else None)

        elbow_path = out_dir / "elbow_kmeans.png"
        plt.figure()
        plt.plot(k_values, inertias, marker="o")
        plt.title("Metodo del codo - KMeans")
        plt.xlabel("K")
        plt.ylabel("Inercia")
        plt.tight_layout()
        plt.savefig(elbow_path, dpi=160)
        plt.close()

        k_opt = int(request.options.get("k_opt", 3))
        kmeans = KMeans(
            n_clusters=k_opt,
            init="k-means++",
            n_init=50,
            max_iter=1000,
            random_state=42,
        )
        kmeans.fit(x_train_scaled)

        train_clusters = kmeans.predict(x_train_scaled)
        test_clusters = kmeans.predict(x_test_scaled)

        mapping = self._map_clusters_to_labels(y_train, train_clusters)
        y_pred_kmeans = np.array([mapping.get(c, 0) for c in test_clusters])
        kmeans_acc = float(accuracy_score(y_test, y_pred_kmeans))

        kmeans_cm_path = out_dir / "confusion_kmeans.png"
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_kmeans,
            display_labels=target_names,
            cmap="Blues",
        )
        plt.title("Matriz de confusion - KMeans")
        plt.tight_layout()
        plt.savefig(kmeans_cm_path, dpi=200)
        plt.close()

        mlp_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        random_state=42,
                        max_iter=2000,
                        early_stopping=True,
                    ),
                ),
            ]
        )
        param_grid = {
            "mlp__hidden_layer_sizes": [(10,), (20,), (30,), (20, 10)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [0.0001, 0.001, 0.01],
            "mlp__learning_rate_init": [0.001, 0.01],
            "mlp__solver": ["adam"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            mlp_pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)
        mlp_best = grid.best_estimator_
        y_pred_mlp = mlp_best.predict(x_test)
        mlp_acc = float(accuracy_score(y_test, y_pred_mlp))

        mlp_cm_path = out_dir / "confusion_mlp.png"
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred_mlp,
            display_labels=target_names,
            cmap="Blues",
        )
        plt.title("Matriz de confusion - MLP")
        plt.tight_layout()
        plt.savefig(mlp_cm_path, dpi=200)
        plt.close()

        image_paths = [elbow_path, kmeans_cm_path, mlp_cm_path]
        expected_images = {"elbow_kmeans.png", "confusion_kmeans.png", "confusion_mlp.png"}
        produced_images = {p.name for p in image_paths}
        missing_images = sorted(expected_images - produced_images)
        if missing_images:
            raise RuntimeError(
                f"Ejercicio3 incompleto: faltan imagenes esperadas: {', '.join(missing_images)}"
            )

        summary = {
            "kmeans": {
                "k_values": k_values,
                "inertias": inertias,
                "silhouettes": silhouettes,
                "k_opt": k_opt,
                "accuracy": kmeans_acc,
                "classification_report": classification_report(
                    y_test,
                    y_pred_kmeans,
                    target_names=target_names,
                    output_dict=True,
                ),
            },
            "mlp": {
                "best_cv_accuracy": float(grid.best_score_),
                "best_params": grid.best_params_,
                "test_accuracy": mlp_acc,
                "classification_report": classification_report(
                    y_test,
                    y_pred_mlp,
                    target_names=target_names,
                    output_dict=True,
                ),
            },
        }
        if "kmeans" not in summary or "mlp" not in summary:
            raise RuntimeError("Ejercicio3 incompleto: faltan resultados de KMeans o MLP")

        return ExerciseResponse(
            type=ExerciseType.ejercicio3,
            summary=summary,
            images=[
                ImageArtifact(
                    name=p.name,
                    path=str(p),
                    url=build_static_url(p),
                    content_base64=image_to_base64(p),
                )
                for p in image_paths
            ],
        )
