import os

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1) Modelo lineal: Logistic Regression
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    # 2) Árbol de decisión
    tree = DecisionTreeClassifier(random_state=42)

    # 3) Random Forest
    rf = RandomForestClassifier(random_state=42)

    # GridSearch para Tree
    grid_tree = GridSearchCV(
        tree,
        param_grid={
            "max_depth": [2, 3, 4, 5, None],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 4, 6]
        },
        cv=cv
    )

    # GridSearch para RF
    grid_rf = GridSearchCV(
        rf,
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3, 4, 5, None],
            "max_features": ["sqrt", "log2", None]
        },
        cv=cv
    )

    modelos = {
        "LogisticRegression": pipe_lr,
        "DecisionTree (GridSearch)": grid_tree,
        "RandomForest (GridSearch)": grid_rf
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        resultados[nombre] = acc

        print("\n" + "=" * 60)
        print(f"MODELO: {nombre}")
        print(f"Accuracy (test): {acc:.4f}")

        if hasattr(modelo, "best_params_"):
            print("Mejores hiperparámetros:", modelo.best_params_)

        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=iris.target_names,
            cmap="Blues"
        )

        plt.title(f"Matriz de Confusión - {nombre}")
        plt.tight_layout()

        nombre_archivo = nombre.replace(" ", "_").replace("(", "").replace(")", "")
        ruta = os.path.join(os.getcwd(), f"confusion_matrix_{nombre_archivo}.png")

        plt.savefig(ruta, dpi=300)
        plt.close()

        print(f"Imagen guardada en: {ruta}")

    print("\n" + "=" * 60)
    print("Resumen Accuracy (test):")
    for k, v in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        print(f"- {k}: {v:.4f}")


if __name__ == "__main__":
    main()
