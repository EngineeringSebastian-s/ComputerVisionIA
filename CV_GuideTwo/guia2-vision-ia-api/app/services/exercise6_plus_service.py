from __future__ import annotations

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops
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


class Exercise6PlusService(ExerciseContract):
    @property
    def exercise_type(self) -> str:
        return ExerciseType.ejercicio6.value

    def _extract_all_features(
            self, root_dir: Path, out_dir: Path, resize: tuple[int, int] = (64, 64), max_samples_per_class: int = 400
    ):
        x_flat, x_rgb, x_hsv, x_tex, y = [], [], [], [], []
        sample_images = []
        sample_labels = []

        classes = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])

        for cls in classes:
            img_paths = list((root_dir / cls).rglob("*"))
            random.seed(42)
            random.shuffle(img_paths)

            loaded_count = 0
            for img_path in img_paths:
                if loaded_count >= max_samples_per_class:
                    break
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue

                # Cargar imagen a color
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                img_resized = cv2.resize(img_bgr, resize, interpolation=cv2.INTER_AREA)
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

                # 1. Flattened (Para la validación del Reto 6 normal)
                x_flat.append(img_gray.flatten().astype(np.float32))

                # 2. RGB Stats
                means_rgb, stds_rgb = cv2.meanStdDev(img_resized)
                x_rgb.append(np.concatenate([means_rgb.flatten(), stds_rgb.flatten()]).astype(np.float32))

                # 3. HSV Stats
                img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
                means_hsv, stds_hsv = cv2.meanStdDev(img_hsv)
                x_hsv.append(np.concatenate([means_hsv.flatten(), stds_hsv.flatten()]).astype(np.float32))

                # 4. Texture Stats (Haralick / GLCM)
                glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                asm = graycoprops(glcm, 'ASM')[0, 0]
                x_tex.append(
                    np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm], dtype=np.float32))

                y.append(cls)
                loaded_count += 1

                # Guardar algunas muestras para la imagen del proceso
                if len(sample_images) < 10 and random.random() < 0.05:
                    sample_images.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
                    sample_labels.append(cls)

        # Generar imagen de muestras visuales del proceso
        if sample_images:
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            fig.suptitle("Muestra del proceso visual: Imágenes cargadas y redimensionadas")
            for i, ax in enumerate(axes.flatten()):
                if i < len(sample_images):
                    ax.imshow(sample_images[i])
                    ax.set_title(sample_labels[i])
                    ax.axis("off")
                else:
                    ax.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / "muestras_proceso_cats_dogs.png", dpi=200)
            plt.close()

        return (
            np.array(x_flat),
            np.array(x_rgb),
            np.array(x_hsv),
            np.array(x_tex),
            np.array(y, dtype=object),
        )

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        project_root = Path(__file__).resolve().parents[2]
        dataset_dir = project_root / "app" / "dataset" / "Cats&Dogs"

        if not dataset_dir.exists():
            raise FileNotFoundError(f"No existe el dataset en {dataset_dir}")

        out_dir = Path("app/output/ejercicio6")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extraer todas las combinaciones de características y guardar la imagen de proceso
        X_flat, X_rgb, X_hsv, X_tex, y = self._extract_all_features(dataset_dir, out_dir, max_samples_per_class=400)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = list(le.classes_)

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        images: list[ImageArtifact] = []
        reports: dict[str, dict] = {}
        comparison_table = []

        # Función auxiliar para evaluar modelos y guardar matrices de confusión
        def evaluate_and_plot(X_data, model_dict, prefix=""):
            X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_enc, test_size=0.3, random_state=42, stratify=y_enc)

            for name, model in model_dict.items():
                full_name = f"{prefix}_{name}" if prefix else name
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                acc = float(accuracy_score(y_te, y_pred))
                reports[full_name] = classification_report(y_te, y_pred, target_names=class_names, output_dict=True)

                cm = confusion_matrix(y_te, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                disp.plot(cmap="Blues")
                plt.title(f"Matriz: {full_name}")
                plt.tight_layout()

                img_path = out_dir / f"confusion_{full_name}.png".replace(" ", "_")
                plt.savefig(img_path, dpi=200)
                plt.close()

                comparison_table.append({
                    "model_or_features": full_name,
                    "accuracy": acc
                })

        # ==========================================
        # PARTE 1: Píxeles Aplanados (Reto 6 Original)
        # ==========================================
        flat_models = {
            "SVM_Baseline": Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=42))]),
            "SVM_Optimized": GridSearchCV(
                Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=42))]),
                param_grid={"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
                cv=cv, n_jobs=-1
            ),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        evaluate_and_plot(X_flat, flat_models, prefix="FLAT")

        # ==========================================
        # PARTE 2: Características Estadísticas (El Extra)
        # ==========================================
        feature_sets = {
            "RGB": X_rgb,
            "RGB_HSV": np.hstack((X_rgb, X_hsv)),
            "TEXTURE": X_tex,
            "RGB_TEXTURE": np.hstack((X_rgb, X_tex)),
            "RGB_HSV_TEXTURE": np.hstack((X_rgb, X_hsv, X_tex)),
        }

        # Evaluamos cada set de características con un SVM estándar para comparar
        stat_models = {"SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])}
        for set_name, X_features in feature_sets.items():
            evaluate_and_plot(X_features, stat_models, prefix=f"EXTRA_{set_name}")

        # ==========================================
        # Gráfico Comparativo Final de todos los enfoques
        # ==========================================
        plt.figure(figsize=(12, 6))
        names = [item["model_or_features"] for item in comparison_table]
        accs = [item["accuracy"] for item in comparison_table]
        bars = plt.barh(names, accs, color='skyblue')
        plt.xlabel("Accuracy")
        plt.title("Comparativa General: Píxeles Aplanados vs Características Estadísticas")
        plt.xlim(0, 1.0)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        for bar in bars:
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center')

        plt.tight_layout()
        comp_img_path = out_dir / "comparativa_general.png"
        plt.savefig(comp_img_path, dpi=200)
        plt.close()

        # Recolectar todas las imágenes para la respuesta
        for p in out_dir.glob("*.png"):
            images.append(
                ImageArtifact(
                    name=p.name,
                    path=str(p),
                    url=build_static_url(p),
                    content_base64=image_to_base64(p),
                )
            )

        best_result = max(comparison_table, key=lambda x: x["accuracy"])

        return ExerciseResponse(
            type=ExerciseType.ejercicio6,
            summary={
                "best_approach": best_result["model_or_features"],
                "best_accuracy": best_result["accuracy"],
                "comparison_table": comparison_table,
                "reports": reports,
            },
            images=images,
        )
