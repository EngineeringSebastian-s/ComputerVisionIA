import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def save_elbow_plot(inertias, k_values, out_path="elbow_kmeans.png"):
    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.title("Método del Codo (K-means) - Inercia vs K")
    plt.xlabel("K (número de clusters)")
    plt.ylabel("Inercia (WCSS)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_confusion_matrix_plot(y_true, y_pred, labels, title, out_path):
    """
    Guarda una matriz de confusión como imagen en el directorio actual.
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def map_clusters_to_labels(y_true, clusters, n_classes=3):
    """
    Asigna clusters -> clases usando Hungarian Algorithm para maximizar coincidencias.
    Funciona mejor cuando K == n_classes.
    """
    cm = confusion_matrix(y_true, clusters)
    cost = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {cluster: label for label, cluster in zip(row_ind, col_ind)}
    return mapping


def apply_mapping(clusters, mapping, fallback_label=0):
    return np.array([mapping.get(c, fallback_label) for c in clusters])


def run_kmeans_elbow_and_eval(X_train_s, y_train, X_test_s, y_test, target_names, k_min=1, k_max=10):
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    # Elbow: inercia para varios K
    for k in k_values:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=500,
            random_state=RANDOM_STATE
        )
        km.fit(X_train_s)
        inertias.append(km.inertia_)

        if k >= 2:
            sil = silhouette_score(X_train_s, km.labels_)
            silhouettes.append(sil)
        else:
            silhouettes.append(np.nan)

    # Guardar elbow
    save_elbow_plot(inertias, k_values, out_path="elbow_kmeans.png")

    # Elegimos K=3 (Iris tiene 3 clases)
    k_opt = 3

    km_final = KMeans(
        n_clusters=k_opt,
        init="k-means++",
        n_init=50,  # hiperparámetro ajustado
        max_iter=1000,  # hiperparámetro ajustado
        random_state=RANDOM_STATE
    )

    km_final.fit(X_train_s)
    train_clusters = km_final.predict(X_train_s)
    test_clusters = km_final.predict(X_test_s)

    # Mapear clusters -> etiquetas usando train
    mapping = map_clusters_to_labels(y_train, train_clusters, n_classes=3)
    y_pred_kmeans = apply_mapping(test_clusters, mapping, fallback_label=0)

    acc = accuracy_score(y_test, y_pred_kmeans)

    print("\n" + "=" * 70)
    print("K-MEANS (NO SUPERVISADO)")
    print("- Se generó la gráfica: elbow_kmeans.png")
    print(f"- K elegido (según codo + conocimiento de 3 clases): {k_opt}")
    print(f"- Accuracy (con mapeo cluster→clase): {acc:.4f}")

    print("\nMatriz de confusión (KMeans):")
    print(confusion_matrix(y_test, y_pred_kmeans))

    print("\nReporte de clasificación (KMeans):")
    print(classification_report(y_test, y_pred_kmeans, target_names=target_names))

    # Guardar matriz de confusión como imagen
    out_cm = os.path.join(os.getcwd(), "confusion_kmeans.png")
    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred_kmeans,
        labels=target_names,
        title="Matriz de Confusión - KMeans (con mapeo clusters→clases)",
        out_path=out_cm
    )
    print(f"- Evidencia guardada: {out_cm}")

    print("\nTabla rápida (K, Inercia, Silhouette en train):")
    for k, iner, sil in zip(k_values, inertias, silhouettes):
        sil_str = "NaN" if np.isnan(sil) else f"{sil:.4f}"
        print(f"  K={k:2d} | Inercia={iner:10.2f} | Silhouette={sil_str}")

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "k_opt": k_opt,
        "accuracy_test": acc
    }


def run_mlp_with_tuning(X_train, y_train, X_test, y_test, target_names):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=2000,
            early_stopping=True
        ))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(10,), (20,), (30,), (20, 10)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__solver": ["adam"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print("MLP (PERCEPTRÓN MULTICAPA - SUPERVISADO)")
    print(f"- Mejor score CV (accuracy): {grid.best_score_:.4f}")
    print(f"- Mejores hiperparámetros: {grid.best_params_}")
    print(f"- Accuracy (test): {acc:.4f}")

    print("\nMatriz de confusión (MLP):")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación (MLP):")
    print(classification_report(y_test, y_pred, target_names=target_names))

    out_cm = os.path.join(os.getcwd(), "confusion_mlp.png")
    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred,
        labels=target_names,
        title="Matriz de Confusión - MLP (mejor modelo GridSearch)",
        out_path=out_cm
    )
    print(f"- Evidencia guardada: {out_cm}")

    return {
        "best_cv_accuracy": float(grid.best_score_),
        "best_params": grid.best_params_,
        "test_accuracy": float(acc)
    }


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Validación: Train/Test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # Escalado para KMeans
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1) K-means + codo + evaluación
    _ = run_kmeans_elbow_and_eval(
        X_train_s, y_train, X_test_s, y_test,
        target_names=target_names,
        k_min=1, k_max=10
    )

    # 2) MLP (supervisado) + tuning + validación CV
    _ = run_mlp_with_tuning(X_train, y_train, X_test, y_test, target_names=target_names)


if __name__ == "__main__":
    main()
