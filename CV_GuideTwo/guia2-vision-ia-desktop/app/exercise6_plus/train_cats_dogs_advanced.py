import os
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
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# --- CONFIGURACIÓN ---
CURRENT_DIR = Path(__file__).resolve().parent
DATASET_PATH = CURRENT_DIR / ".."/ "exercise6" /"Cats&Dogs"


def extract_features_from_images(root_dir, resize=(64, 64), max_samples=400):
    root_path = Path(root_dir)
    x_flat, x_rgb, x_hsv, x_tex, y = [], [], [], [], []

    if not root_path.exists():
        print(f"Error: No se encuentra {root_dir}")
        return None

    classes = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    print(f"Clases encontradas: {classes}")

    for cls in classes:
        img_paths = list((root_path / cls).rglob("*"))
        random.seed(42)
        random.shuffle(img_paths)

        loaded = 0
        for img_path in img_paths:
            if loaded >= max_samples:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: continue

            img_resized = cv2.resize(img_bgr, resize, interpolation=cv2.INTER_AREA)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # 1. Flatten
            x_flat.append(img_gray.flatten().astype(np.float32))

            # 2. RGB
            m_rgb, s_rgb = cv2.meanStdDev(img_resized)
            x_rgb.append(np.concatenate([m_rgb.flatten(), s_rgb.flatten()]))

            # 3. HSV
            img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
            m_hsv, s_hsv = cv2.meanStdDev(img_hsv)
            x_hsv.append(np.concatenate([m_hsv.flatten(), s_hsv.flatten()]))

            # 4. Textura (Haralick)
            glcm = graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            asm = graycoprops(glcm, 'ASM')[0, 0]

            x_tex.append(np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm]))

            y.append(cls)
            loaded += 1

    return (
        np.array(x_flat, dtype=np.float32),
        np.array(x_rgb, dtype=np.float32),
        np.array(x_hsv, dtype=np.float32),
        np.array(x_tex, dtype=np.float32),
        np.array(y, dtype=object)
    )


def evaluate_models(X_train, X_test, y_train, y_test, class_names, models_dict, prefix):
    results = {}
    for name, model in models_dict.items():
        full_name = f"{prefix} | {name}"
        print(f"Entrenando {full_name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[full_name] = acc
        print(f"   -> Accuracy: {acc:.4f}")

        # Guardar matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues")
        plt.title(f"{full_name}")
        plt.tight_layout()
        nombre_archivo = full_name.replace(" | ", "_").replace(" ", "")
        plt.savefig(CURRENT_DIR / f"cm_{nombre_archivo}.png", dpi=200)
        plt.close()

    return results


def main():
    print("Extrayendo características de imágenes (Flatten, RGB, HSV, GLCM)...")
    data = extract_features_from_images(DATASET_PATH, max_samples=400)
    if not data: return

    X_flat, X_rgb, X_hsv, X_tex, y = data

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    print(f"Imágenes totales: {len(y)}")

    # Divisiones para cada set
    split_args = {"test_size": 0.30, "random_state": 42, "stratify": y_enc}

    X_flat_tr, X_flat_te, y_tr, y_te = train_test_split(X_flat, y_enc, **split_args)

    # --- PARTE 1: Reto Clásico (Aplanado) ---
    print("\n--- 1. EVALUANDO PÍXELES APLANADOS (Baseline Reto 6) ---")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    flat_models = {
        "SVM_Baseline": Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=42))]),
        "SVM_Optimized": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(random_state=42))]),
            param_grid={"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]},
            cv=cv, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    resultados_flat = evaluate_models(X_flat_tr, X_flat_te, y_tr, y_te, class_names, flat_models, "FLAT")

    # --- PARTE 2: Características Estadísticas (El Extra) ---
    print("\n--- 2. EVALUANDO CARACTERÍSTICAS ESTADÍSTICAS E HÍBRIDAS ---")

    feature_sets = {
        "RGB": X_rgb,
        "RGB_HSV": np.hstack((X_rgb, X_hsv)),
        "TEXTURE": X_tex,
        "RGB_TEXTURE": np.hstack((X_rgb, X_tex)),
        "RGB_HSV_TEXTURE": np.hstack((X_rgb, X_hsv, X_tex)),
    }

    # Usaremos un SVM RBF estándar para evaluar estas características
    stat_model = {"SVM_RBF": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", random_state=42))])}

    resultados_extra = {}
    for set_name, X_features in feature_sets.items():
        X_feat_tr, X_feat_te, _, _ = train_test_split(X_features, y_enc, **split_args)
        res = evaluate_models(X_feat_tr, X_feat_te, y_tr, y_te, class_names, stat_model, f"EXTRA_{set_name}")
        resultados_extra.update(res)

    # --- RESUMEN FINAL Y GRÁFICA ---
    todos_resultados = {**resultados_flat, **resultados_extra}

    print("\n" + "=" * 60)
    print("RANKING FINAL DE EXACTITUD (ACCURACY):")
    for k, v in sorted(todos_resultados.items(), key=lambda item: item[1], reverse=True):
        print(f"- {v:.4f} : {k}")

    # Generar gráfica de barras comparativa
    plt.figure(figsize=(10, 6))
    nombres = list(todos_resultados.keys())
    valores = list(todos_resultados.values())

    y_pos = np.arange(len(nombres))
    bars = plt.barh(y_pos, valores, align='center', color='coral')
    plt.yticks(y_pos, nombres)
    plt.xlabel('Accuracy')
    plt.title('Comparativa General de Modelos y Características (Cats vs Dogs)')
    plt.xlim(0, 1.0)

    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.4f}', va='center')

    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "comparativa_final_cats_dogs.png", dpi=300)
    print("\n-> Gráfico resumen guardado como 'comparativa_final_cats_dogs.png'")


if __name__ == "__main__":
    main()