# CV Guide Two - Vision por Computadora + Machine Learning

Este proyecto implementa una capa de IA escalable para ejercicios de visión por computadora y aprendizaje automático, expuesta mediante una API en FastAPI y consumida por un frontend en Angular.

La aplicación permite ejecutar distintos ejercicios (1, 3 y 5 implementados), visualizar métricas por modelo y mostrar imágenes de evidencia generadas por los experimentos.

---

## Resumen del proyecto

La idea central del proyecto es comparar modelos de Machine Learning sobre tareas de clasificación y agrupamiento, no solo desde la exactitud final, sino también desde métricas por clase, estabilidad y calidad de la representación de datos.

A nivel técnico:

- **Backend (FastAPI)**: ejecuta los ejercicios, entrena/evalúa modelos, genera reportes e imágenes, y expone resultados en JSON.
- **Frontend (Angular)**: consume la API y presenta resultados por bloques (resumen, métricas por modelo, tablas comparativas e imágenes).
- **Contenedores (Docker + Compose)**: facilitan ejecución local integrada.

En términos analíticos, el proyecto confirma tres ideas importantes:

1. Un modelo más complejo no siempre gana.
2. La ingeniería de características puede impactar más que cambiar de algoritmo.
3. La interpretación por clase es clave: la métrica global por sí sola no cuenta toda la historia.

---

## Stack tecnológico

- **API**: FastAPI + scikit-learn + matplotlib + OpenCV + scikit-image
- **Frontend**: Angular 21
- **Despliegue local**: Docker + Docker Compose
- **Salida de artefactos**: imágenes y reportes en `/api/app/output`

---

## Arquitectura funcional

- `api/`
  - Controlador de ejecución de ejercicios.
  - Servicios por ejercicio (`exercise1_service.py`, `exercise3_service.py`, `exercise5_service.py`).
  - Exposición de archivos estáticos para imágenes generadas.
- `frontend/guia2-vision-ia/`
  - Menú principal de ejercicios.
  - Vista de resultados por ejercicio con presentación agrupada.
  - Servicio HTTP hacia FastAPI.

---

## Análisis de resultados - Ejercicio 1

### Objetivo
Comparar clasificadores supervisados sobre Iris:

- Regresión Logística
- Árbol de Decisión
- Random Forest

### Resultado principal
El mejor modelo fue **DecisionTree** con **accuracy = 0.9778**.

- LogisticRegression: 0.9111
- DecisionTree: 0.9778
- RandomForest: 0.9111

- **Setosa** fue separada perfectamente por todos los modelos (métricas de 1.0).
- Los errores se concentraron en **versicolor vs virginica**, que son clases más solapadas.
- El **Árbol de Decisión** rindió mejor porque modela fronteras no lineales mediante reglas jerárquicas.
- RandomForest no superó al árbol individual en esta corrida, probablemente por configuración y tamaño del dataset.

### Conclusión
Para esta partición/configuración, **DecisionTree** fue el clasificador más adecuado en Iris.

---

## Análisis de resultados - Ejercicio 3

### Objetivo
Comparar un enfoque no supervisado vs supervisado en Iris:

- **K-means** (clustering)
- **MLP** (clasificación supervisada)

### Hallazgos de K-means

- El codo sugiere **K=3** (coincide con 3 especies).
- La silueta máxima aparece en K=2, mostrando que la métrica geométrica no siempre coincide con clases reales.
- Accuracy mapeado a etiquetas: **0.7556**.
- Setosa se agrupa muy bien; los errores vuelven a concentrarse en versicolor/virginica.

### Hallazgos de MLP

- Mejores hiperparámetros: `tanh`, `alpha=0.0001`, capa oculta `(20)`, `learning_rate_init=0.01`, `adam`.
- Best CV accuracy: **0.9048**
- Test accuracy: **0.8444**

### Comparación

- MLP (supervisado) > K-means (no supervisado) en exactitud.
- Aun así, en este dataset pequeño no necesariamente supera a los mejores modelos de árbol del ejercicio 1.

### Conclusión

- **K-means** es útil para explorar estructura.
- **MLP** es mejor para clasificar cuando hay etiquetas.
- El ajuste de hiperparámetros es determinante.

---

## Análisis de resultados - Ejercicio 5

### Objetivo
Clasificar escenas (`coast`, `forest`, `highway`) evaluando:

- Diferentes **conjuntos de características**:
  - RGB
  - RGB+HSV
  - Texture
  - RGB+Texture
  - RGB+HSV+Texture
- Múltiples modelos (SVM, RandomForest, GradBoost, MLP, etc.).

### Resultado global más fuerte
Mejor combinación:

- **RGB + Texture** con **SVM-RBF**
- Accuracy: **0.9201**
- F1 macro: **0.9163**

### Lectura comparativa por features

- RGB: bueno, pero limitado.
- RGB+HSV: muy similar a RGB (sin mejora clara).
- Texture: mejora fuerte sobre color puro.
- RGB+Texture: mejor equilibrio y mejor resultado global.
- RGB+HSV+Texture: excelente, pero no supera de forma clara a RGB+Texture.

### Lectura comparativa por modelos

Más robustos en general:

- **SVM-RBF**
- **RandomForest**
- **GradientBoosting**

Más débiles/inestables en varios escenarios:

- NaiveBayes
- DecisionTree (árbol único)

### Comportamiento por clase

- **forest** fue la clase más fácil.
- **highway** la más difícil.
- **coast** en zona intermedia.

### Conclusión
La mejora principal vino de la **representación de entrada** (features), especialmente al combinar **color + textura**.

---

## Ejecución rápida con Docker

Desde la raíz del proyecto:

```bash
docker compose up --build -d
```

Servicios esperados:

- API: `http://localhost:8000`
- Frontend: `http://localhost:4200`

---

## Estado actual

- Frontend con visualización específica para estos ejercicios.
- Exposición de artefactos (imágenes) vía archivos estáticos.
