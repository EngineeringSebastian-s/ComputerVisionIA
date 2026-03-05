from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.contracts.exercise_contract import ExerciseContract
from app.schemas.execution import ExerciseRequest, ExerciseResponse, ExerciseType, ImageArtifact
from app.utils.image_utils import image_to_base64
from app.utils.static_url import build_static_url


class Exercise5Service(ExerciseContract):
    @property
    def exercise_type(self) -> str:
        return ExerciseType.ejercicio5.value

    @staticmethod
    def _rgb_stats(img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        feats = []
        for c in range(3):
            channel = img_rgb[:, :, c]
            feats.extend([channel.mean(), channel.std()])
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _hsv_stats(img_bgr: np.ndarray) -> np.ndarray:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        feats = []
        for c in range(3):
            channel = img_hsv[:, :, c]
            feats.extend([channel.mean(), channel.std()])
        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _glcm_texture(img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_32 = (gray // 8).astype(np.uint8)
        glcm = graycomatrix(
            gray_32,
            distances=[1, 2, 3],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=32,
            symmetric=True,
            normed=True,
        )
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
        feats = []
        for p in props:
            val = graycoprops(glcm, p)
            feats.extend([val.mean(), val.std()])
        return np.array(feats, dtype=np.float32)

    def _extract_features(self, img_bgr: np.ndarray, mode: str) -> np.ndarray:
        parts = []
        if "rgb" in mode:
            parts.append(self._rgb_stats(img_bgr))
        if "hsv" in mode:
            parts.append(self._hsv_stats(img_bgr))
        if "texture" in mode:
            parts.append(self._glcm_texture(img_bgr))
        if not parts:
            raise ValueError(f"Modo de features invalido: {mode}")
        return np.concatenate(parts)

    def _load_3scenes(
        self,
        root_dir: Path,
        mode: str,
        resize: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        classes = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])
        for cls in classes:
            for img_path in (root_dir / cls).rglob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                x.append(self._extract_features(img, mode=mode))
                y.append(cls)

        if not x:
            raise RuntimeError("No se cargaron imagenes para ejercicio5")
        return np.array(x, dtype=np.float32), np.array(y, dtype=object)

    @staticmethod
    def _get_models() -> dict[str, object]:
        return {
            "LogReg": Pipeline(
                [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]
            ),
            "SVM-RBF": Pipeline(
                [("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=42))]
            ),
            "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
            "NaiveBayes": GaussianNB(),
            "LDA": Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
            "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
            "GradBoost": GradientBoostingClassifier(random_state=42),
            "MLP": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            alpha=0.001,
                            learning_rate_init=0.001,
                            max_iter=3000,
                            early_stopping=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

    def _evaluate_mode(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mode_tag: str,
        out_dir: Path,
    ) -> tuple[dict, Path, Path]:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = list(le.classes_)
        label_idx = list(range(len(class_names)))

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y_enc,
            test_size=0.3,
            random_state=42,
            stratify=y_enc,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rows = []
        best_name = ""
        best_model = None
        best_f1 = -1.0
        model_names: list[str] = []

        for name, model in self._get_models().items():
            cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            acc = float(accuracy_score(y_test, y_pred))
            f1m = float(f1_score(y_test, y_pred, average="macro"))

            rows.append(
                {
                    "model": name,
                    "cv_f1_macro_mean": float(cv_scores.mean()),
                    "cv_f1_macro_std": float(cv_scores.std()),
                    "test_accuracy": acc,
                    "test_f1_macro": f1m,
                }
            )
            model_names.append(name)

            if f1m > best_f1:
                best_f1 = f1m
                best_name = name
                best_model = model

        results_df = pd.DataFrame(rows).sort_values(by="test_f1_macro", ascending=False)
        expected_models = list(self._get_models().keys())
        missing_models = sorted(set(expected_models) - set(model_names))
        if missing_models:
            raise RuntimeError(
                f"Ejercicio5 incompleto en {mode_tag}: faltan resultados para modelos: {', '.join(missing_models)}"
            )

        csv_path = out_dir / f"{mode_tag}_results.csv"
        results_df.to_csv(csv_path, index=False)

        y_pred_best = best_model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred_best, labels=label_idx)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(xticks_rotation=45)
        plt.title(f"Mejor modelo: {best_name} | {mode_tag}")
        plt.tight_layout()

        img_path = out_dir / f"{mode_tag}_confusion_best.png"
        plt.savefig(img_path, dpi=160)
        plt.close()

        report = classification_report(y_test, y_pred_best, target_names=class_names, output_dict=True)
        txt_path = out_dir / f"{mode_tag}_best_report.txt"
        txt_path.write_text(
            f"FEATURE SET: {mode_tag}\nMejor modelo: {best_name}\n\n"
            + classification_report(y_test, y_pred_best, target_names=class_names),
            encoding="utf-8",
        )

        summary = {
            "feature_set": mode_tag,
            "best_model": best_name,
            "best_test_f1_macro": best_f1,
            "table": rows,
            "classification_report_best": report,
            "csv_path": str(csv_path),
            "report_path": str(txt_path),
            "csv_url": build_static_url(csv_path),
            "report_url": build_static_url(txt_path),
            "best_confusion_path": str(img_path),
            "best_confusion_url": build_static_url(img_path),
        }

        return summary, img_path, txt_path

    def execute(self, request: ExerciseRequest) -> ExerciseResponse:
        project_root = Path(__file__).resolve().parents[2]
        dataset_dir = project_root / "EJERCICIO-5" / "3scenes"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"No existe dataset en {dataset_dir}")

        out_dir = Path("app/output/ejercicio5")
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_modes = request.options.get(
            "feature_modes",
            ["rgb", "rgb+hsv", "texture", "rgb+texture", "rgb+hsv+texture"],
        )
        mode_map = {
            "rgb": "RGB",
            "rgb+hsv": "RGB_HSV",
            "texture": "TEXTURE",
            "rgb+texture": "RGB_TEXTURE",
            "rgb+hsv+texture": "RGB_HSV_TEXTURE",
        }

        resize_raw = request.options.get("resize", [128, 128])
        resize = (int(resize_raw[0]), int(resize_raw[1]))

        summaries = []
        images: list[ImageArtifact] = []

        for mode in feature_modes:
            if mode not in mode_map:
                raise ValueError(f"feature_mode invalido: {mode}")

            mode_tag = mode_map[mode]
            x, y = self._load_3scenes(dataset_dir, mode=mode, resize=resize)
            summary, best_img_path, _ = self._evaluate_mode(x, y, mode_tag, out_dir)
            summaries.append(summary)
            # Se expone solo la matriz de confusion del mejor modelo por modo
            # (comportamiento equivalente al script original).
            images.append(
                ImageArtifact(
                    name=best_img_path.name,
                    path=str(best_img_path),
                    url=build_static_url(best_img_path),
                    content_base64=image_to_base64(best_img_path),
                )
            )

        expected_max_images = len(feature_modes)
        if len(images) > expected_max_images:
            raise RuntimeError(
                f"Ejercicio5 inconsistente: se esperaban maximo {expected_max_images} imagenes, "
                f"obtenidas {len(images)}"
            )

        return ExerciseResponse(
            type=ExerciseType.ejercicio5,
            summary={"modes": summaries},
            images=images,
        )
