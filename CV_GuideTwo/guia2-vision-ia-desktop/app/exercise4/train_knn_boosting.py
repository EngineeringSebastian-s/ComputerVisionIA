import os

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Cargar el dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. Dividir los datos en entrenamiento y prueba (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 3. Configurar validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4. Definir modelos y búsqueda de hiperparámetros

    # --- Modelo 1: K-Nearest Neighbors (KNN) ---
    # KNN es muy sensible a la escala de las características, usamos un Pipeline
    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    grid_knn = GridSearchCV(
        pipe_knn,
        param_grid={
            "knn__n_neighbors": [3, 5, 7, 9],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        },
        cv=cv,
        scoring="accuracy"
    )

    # --- Modelo 2: Gradient Boosting Classifier ---
    gb = GradientBoostingClassifier(random_state=42)
    grid_gb = GridSearchCV(
        gb,
        param_grid={
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5]
        },
        cv=cv,
        scoring="accuracy"
    )

    # Diccionario para iterar fácilmente
    modelos = {
        "KNN": grid_knn,
        "GradientBoosting": grid_gb
    }

    resultados = {}

    # 5. Entrenamiento, evaluación y guardado de resultados
    for nombre, modelo in modelos.items():
        # Entrenar el modelo con GridSearch
        modelo.fit(X_train, y_train)

        # Predecir con el conjunto de prueba
        y_pred = modelo.predict(X_test)

        # Calcular Accuracy
        acc = accuracy_score(y_test, y_pred)
        resultados[nombre] = acc

        print("\n" + "=" * 60)
        print(f"MODELO: {nombre}")
        print(f"Accuracy (test): {acc:.4f}")
        print(f"Mejor score CV (train):  {modelo.best_score_:.4f}")

        if hasattr(modelo, "best_params_"):
            print("Mejores hiperparámetros:", modelo.best_params_)

        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

        # Generar la gráfica de la matriz de confusión
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=iris.target_names,
            cmap="Blues"
        )

        plt.title(f"Matriz de Confusión - {nombre}")
        plt.tight_layout()

        # Guardar la imagen en el directorio actual
        nombre_archivo = nombre.replace(" ", "_")
        ruta = os.path.join(os.getcwd(), f"confusion_matrix_{nombre_archivo}.png")

        plt.savefig(ruta, dpi=300)
        plt.close()

        print(f"- Imagen guardada en: {ruta}")

    # 6. Resumen final
    print("\n" + "=" * 60)
    print("Resumen Accuracy (test):")
    for k, v in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        print(f"- {k}: {v:.4f}")


if __name__ == "__main__":
    main()
