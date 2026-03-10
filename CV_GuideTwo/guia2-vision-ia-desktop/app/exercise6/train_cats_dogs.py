import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
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

# --- CONFIGURACIÓN DE RUTA LOCAL ---
# Esto detecta la carpeta donde está este script y busca "Cats&Dogs" justo a su lado
CURRENT_DIR = Path(__file__).resolve().parent
DATASET_PATH = CURRENT_DIR / "Cats&Dogs"


def load_cats_dogs(root_dir, resize=(64, 64), max_samples=600):
    root_path = Path(root_dir)
    x, y = [], []

    if not root_path.exists():
        print(f"Error: La ruta {root_dir} no existe. Verifica que la carpeta 'Cats&Dogs' esté junto a este .py")
        return None, None

    # Leer las subcarpetas (ej. "Cats" y "Dogs")
    classes = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
    print(f"Clases encontradas: {classes}")

    for cls in classes:
        img_paths = list((root_path / cls).rglob("*"))
        # Mezclar para tomar una muestra representativa y no solo las primeras
        random.seed(42)
        random.shuffle(img_paths)

        loaded = 0
        for img_path in img_paths:
            # Límite de imágenes para no saturar la memoria RAM en ejecución local
            if loaded >= max_samples:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            # Cargar en escala de grises para reducir el tiempo de entrenamiento de la SVM
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Redimensionar y aplanar a un vector 1D
            img_resized = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            img_flattened = img_resized.flatten().astype(np.float32)

            x.append(img_flattened)
            y.append(cls)
            loaded += 1

    return np.array(x, dtype=np.float32), np.array(y, dtype=object)


def main():
    print("Cargando imágenes (escala de grises, redimensionadas a 64x64 y aplanadas)...")
    X, y = load_cats_dogs(DATASET_PATH, max_samples=600)

    if X is None:
        return

    print(f"Total imágenes cargadas: {X.shape[0]} con {X.shape[1]} características cada una.")

    # Codificar las etiquetas (Cats -> 0, Dogs -> 1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    # Dividir datos (70% entrenamiento, 30% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.30, random_state=42, stratify=y_enc
    )

    # Configurar validación cruzada
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Definir Modelos (Base, Optimizado y Alternativo)
    modelos = {
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

    resultados = {}

    # Entrenamiento y evaluación
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        resultados[nombre] = acc

        print(f"Accuracy (test): {acc:.4f}")

        if hasattr(modelo, "best_params_"):
            print("Mejores hiperparámetros encontrados:", modelo.best_params_)

        print("Reporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Generar y guardar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="Blues")
        plt.title(f"Matriz de Confusion - {nombre}")
        plt.tight_layout()

        ruta_img = CURRENT_DIR / f"confusion_matrix_{nombre}.png"
        plt.savefig(ruta_img, dpi=300)
        plt.close()
        print(f"-> Imagen guardada en: {ruta_img}")

    # Resumen final ordenado por mejor Accuracy
    print("\n" + "=" * 60)
    print("Resumen de Accuracy (test) [Perros vs Gatos]:")
    for k, v in sorted(resultados.items(), key=lambda item: item[1], reverse=True):
        print(f"- {k}: {v:.4f}")


if __name__ == "__main__":
    main()
