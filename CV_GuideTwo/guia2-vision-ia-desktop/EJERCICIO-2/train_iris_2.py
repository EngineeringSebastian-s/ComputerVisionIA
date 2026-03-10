import os
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


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

    # --- Modelo 1: Naive Bayes (GaussianNB) ---
    nb = GaussianNB()
    grid_nb = GridSearchCV(
        nb,
        param_grid={
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        },
        cv=cv,
        scoring="accuracy"
    )

    # --- Modelo 2: Máquinas de Soporte Vectorial (SVM) ---
    # SVM es muy sensible a la escala de las características, por lo que usamos un Pipeline
    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(random_state=42))
    ])

    grid_svm = GridSearchCV(
        pipe_svm,
        param_grid={
            "svm__C": [0.1, 1, 10, 100],
            "svm__kernel": ["linear", "rbf", "poly"],
            "svm__gamma": ["scale", "auto"]
        },
        cv=cv,
        scoring="accuracy"
    )

    # Diccionario para iterar fácilmente
    modelos = {
        "NaiveBayes": grid_nb,
        "SVM": grid_svm
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