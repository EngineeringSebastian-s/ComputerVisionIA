# scenes.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from skimage.feature import graycomatrix, graycoprops

RANDOM_STATE = 42


# ---------------------------
# Feature extraction
# ---------------------------

def rgb_stats(img_bgr: np.ndarray) -> np.ndarray:
    """Media y desviación estándar por canal RGB (6 features)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    feats = []
    for c in range(3):  # R,G,B
        channel = img_rgb[:, :, c]
        feats.append(channel.mean())
        feats.append(channel.std())
    return np.array(feats, dtype=np.float32)


def hsv_stats(img_bgr: np.ndarray) -> np.ndarray:
    """Media y desviación estándar por canal HSV (6 features)."""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    feats = []
    for c in range(3):  # H,S,V
        channel = img_hsv[:, :, c]
        feats.append(channel.mean())
        feats.append(channel.std())
    return np.array(feats, dtype=np.float32)


def glcm_texture(img_bgr: np.ndarray) -> np.ndarray:
    """
    Texturas tipo Haralick (basadas en GLCM).
    Calcula propiedades y promedia sobre varias distancias/ángulos.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Reducir niveles para que GLCM sea más estable/rápido (0..31)
    gray_32 = (gray // 8).astype(np.uint8)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        gray_32,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        val = graycoprops(glcm, p)  # shape: (len(distances), len(angles))
        feats.append(val.mean())
        feats.append(val.std())

    return np.array(feats, dtype=np.float32)


def extract_features(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    """
    mode:
      - "rgb"
      - "rgb+hsv"
      - "rgb+texture"
      - "rgb+hsv+texture"
      - "texture"
    """
    parts = []
    if "rgb" in mode:
        parts.append(rgb_stats(img_bgr))
    if "hsv" in mode:
        parts.append(hsv_stats(img_bgr))
    if "texture" in mode:
        parts.append(glcm_texture(img_bgr))

    if not parts:
        raise ValueError(f"Modo de features inválido: {mode}")

    return np.concatenate(parts)


# ---------------------------
# Dataset loader
# ---------------------------

def load_3scenes(root_dir: str, mode: str, resize=(128, 128)) -> tuple[np.ndarray, np.ndarray, list[str]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe la ruta: {root_dir}")

    X, y = [], []
    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])

    if not class_names:
        raise RuntimeError("No encontré subcarpetas de clases dentro de 3scenes.")

    for cls in class_names:
        cls_dir = root / cls
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            if resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

            feats = extract_features(img, mode=mode)
            X.append(feats)
            y.append(cls)  # etiqueta en texto (se codifica a int en evaluate_models)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=object)

    if len(X) == 0:
        raise RuntimeError("No se cargaron imágenes. Revisa que haya .jpg/.png dentro de las carpetas.")

    return X, y, class_names


# ---------------------------
# Models (9 típicos)
# ---------------------------

def get_models():
    # Nota: algunos necesitan escalado (LR, SVC, KNN, MLP, LDA).
    # Para simplificar, usamos Pipeline con StandardScaler en la mayoría.
    models = {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7)),
        ]),
        "NaiveBayes": GaussianNB(),
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE),
        "GradBoost": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=3000,
                early_stopping=True,  # OK ahora porque y será numérico (LabelEncoder)
                random_state=RANDOM_STATE
            )),
        ]),
    }
    return models


# ---------------------------
# Training + Validation
# ---------------------------

def evaluate_models(X, y, class_names, out_prefix: str):
    """
    Evalúa 9 modelos con CV (F1 macro) y test (accuracy/F1 macro).
    Guarda CSV, matriz de confusión y reporte TXT
    dentro de una carpeta organizada por feature set.
    """

    # -----------------------------
    # Crear carpeta organizada
    # -----------------------------
    BASE_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = BASE_DIR / "resultados" / out_prefix
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- FIX: labels string -> int ----
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    labels_idx = list(range(len(class_names)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.30, random_state=RANDOM_STATE, stratify=y_enc
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_name, best_model, best_f1 = None, None, -1.0

    models = get_models()

    for name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        results.append({
            "model": name,
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1m),
        })

        if f1m > best_f1:
            best_f1 = f1m
            best_name = name
            best_model = model

    # -----------------------------
    # Guardar CSV resultados
    # -----------------------------
    df_res = pd.DataFrame(results).sort_values(by="test_f1_macro", ascending=False)
    csv_path = RESULTS_DIR / f"{out_prefix}_results.csv"
    df_res.to_csv(csv_path, index=False)

    # -----------------------------
    # Guardar matriz de confusión
    # -----------------------------
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=labels_idx)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(f"Mejor modelo: {best_name} | {out_prefix}")
    plt.tight_layout()

    png_path = RESULTS_DIR / f"{out_prefix}_confusion_best.png"
    plt.savefig(png_path, dpi=160)
    plt.close()

    # -----------------------------
    # Guardar reporte TXT
    # -----------------------------
    report = classification_report(y_test, y_pred_best, target_names=class_names)
    txt_path = RESULTS_DIR / f"{out_prefix}_best_report.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"FEATURE SET: {out_prefix}\n")
        f.write(f"Mejor modelo: {best_name}\n\n")
        f.write(report)

    # -----------------------------
    # Print consola
    # -----------------------------
    print("\n" + "=" * 80)
    print(f"FEATURE SET: {out_prefix}")
    print(df_res.to_string(index=False))
    print(f"\n Guardado en carpeta: {RESULTS_DIR}")
    print(f"  - CSV: {csv_path.name}")
    print(f"  - PNG: {png_path.name}")
    print(f"  - TXT: {txt_path.name}")

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR / "3scenes"

    print("Dataset dir:", DATASET_DIR)
    print("Existe:", DATASET_DIR.exists())

    feature_modes = [
        ("rgb", "RGB"),
        ("rgb+hsv", "RGB_HSV"),
        ("texture", "TEXTURE"),
        ("rgb+texture", "RGB_TEXTURE"),
        ("rgb+hsv+texture", "RGB_HSV_TEXTURE"),
    ]

    for mode, tag in feature_modes:
        X, y, class_names = load_3scenes(DATASET_DIR, mode=mode, resize=(128, 128))
        evaluate_models(X, y, class_names, out_prefix=tag)

    print("\nListo. Ya tienes resultados comparables por conjunto de características.")


if __name__ == "__main__":
    main()