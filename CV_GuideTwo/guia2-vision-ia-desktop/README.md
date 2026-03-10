# Informe de Resultados: Aprendizaje Automático y Visión por Computador

## 1. Reto “Clasificación de Flores parte 1”

**Implementación y Resultados:**
Se utilizó el dataset Iris original. Se dividieron los datos (70% entrenamiento, 30% prueba) y se entrenaron tres modelos:

* **Regresión Logística:** Exactitud de 0.9111
* **Árbol de Decisión:** Exactitud de 0.9778
* **Random Forest:** Exactitud de 0.9111

**Optimización y Fundamentación Matemática:**
Para mejorar los resultados, se utilizó validación cruzada (`GridSearchCV`). El Árbol de Decisión destacó seleccionando hiperparámetros óptimos que evitaron el sobreajuste. Matemáticamente, el árbol de decisión utiliza la ganancia de información (basada en la entropía) o el índice Gini para realizar las divisiones de las características. La pureza de los nodos (Gini) se calcula como: $Gini = 1 - \sum (p_i)^2$, donde $p_i$ es la probabilidad de un elemento perteneciente a una clase específica.

**Conclusiones:**
El **Árbol de Decisión** obtuvo el mejor rendimiento (cerca del 98%). En este dataset, que posee características muy bien definidas espacialmente para la clase *Setosa* y algo mezcladas para *Versicolor* y *Virginica*, las reglas ortogonales del árbol fueron suficientes para trazar límites de decisión casi perfectos, sin necesidad de usar ensambles más complejos como Random Forest.

---

## 2. Reto “Clasificación de Flores parte 2”

**Implementación y Resultados:**
Se exploraron métodos estadísticos y basados en márgenes espaciales:

* **Naive Bayes:** Exactitud de 0.9111
* **Máquinas de Soporte Vectorial (SVM):** Exactitud de 0.9333

**Optimización y Fundamentación Matemática:**
Se realizó una búsqueda en malla para la SVM, encontrando como mejores parámetros un parámetro de penalización de error $C=100$, un `gamma=scale` y un kernel `linear`. La fundamentación de la SVM se basa en encontrar el hiperplano óptimo que maximiza el margen entre las clases de flores. La minimización principal obedece a $\frac{1}{2} ||w||^2 + C \sum \xi_i$, donde $w$ es el vector normal al hiperplano y $\xi_i$ representa la variable de holgura. Para Naive Bayes, se optimizó el parámetro de suavizado de varianza (`var_smoothing = 1e-09`), apoyándose en el Teorema de Bayes asumiendo distribuciones gaussianas e independencia entre los sépalos y pétalos.

**Conclusiones:**
El **SVM lineal** fue superior porque el algoritmo determinó que, penalizando severamente el error ($C=100$), las fronteras de decisión de este conjunto de datos se pueden resolver linealmente en el espacio de características, sin necesidad de proyectar a múltiples dimensiones usando un kernel RBF.

---

## 3. Reto “Clasificación de Flores parte 3”

**Implementación y Resultados:**
Se comparó un enfoque de aprendizaje no supervisado frente a uno supervisado:

* **K-Means (No Supervisado):** Exactitud aproximada de 0.7556
* **Perceptrón Multicapa (MLP) (Supervisado):** Exactitud de 0.8444

**Optimización y Fundamentación Matemática:**
Para K-Means, se analizó la inercia utilizando el "Método del Codo". K-Means minimiza la varianza intra-cluster iterando sobre los centroides $\mu_j$ de la ecuación: $\sum_{i=1}^n \min_{\mu_j} ||x_i - \mu_j||^2$. El codo mostró el punto óptimo en $K=3$. Para la red neuronal artificial (MLP), la optimización con propagación hacia atrás actualiza los pesos de las capas ocultas utilizando el descenso de gradiente.

**Conclusiones:**
Al ser K-Means un algoritmo de agrupamiento aglomerativo espacial que no conoce las etiquetas reales (solo mide la distancia geométrica), tiende a confundir fuertemente las flores Versicolor y Virginica. El Perceptrón Multicapa **supera el rendimiento**, validando la teoría de que proporcionar etiquetas durante el entrenamiento (supervisión) guía el descenso del gradiente para capturar relaciones no lineales complejas.

---

## 4. Reto “Clasificación de Flores parte 4”

**Implementación y Resultados:**
Se introdujeron dos nuevos algoritmos:

* **K-Nearest Neighbors (KNN):** Exactitud de 0.9333
* **Gradient Boosting Classifier:** Exactitud de 0.9778

**Optimización y Fundamentación Matemática:**
El mejor KNN usó 7 vecinos (`n_neighbors=7`), peso por distancia y métrica euclidiana. Matemáticamente, evalúa la distancia espacial: $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$ y da mayor poder de voto a los puntos más cercanos. El **Gradient Boosting** alcanzó la mejor exactitud con una tasa de aprendizaje de 0.01 y 50 estimadores. Su fundamento radica en entrenar árboles de decisión iterativos donde cada nuevo árbol minimiza la función de pérdida residual del modelo anterior.

**Conclusiones:**
El método de ensamble **Gradient Boosting** demuestra que construir modelos secuenciales que corrigen los errores de sus predecesores es una de las estrategias de aprendizaje automático clásico más poderosas, igualando el récord más alto de la práctica (98%).

---

## 5. Reto “Clasificación de escenas: costas, bosques y autopistas”

**Implementación y Resultados:**
Se usó el dataset de 3Scenes y se construyeron 5 espacios de características. Se evaluaron con F1-Score (macro) debido a la naturaleza multiclase del problema:

1. **RGB:** Mejor modelo SVM-RBF (F1: 0.8289)
2. **RGB + HSV:** Mejor modelo SVM-RBF (F1: 0.8274)
3. **Textura (Descriptores Haralick GLCM):** Mejor modelo Random Forest (F1: 0.8772)
4. **RGB + Textura:** Mejor modelo SVM-RBF (F1: 0.9163)
5. **RGB + HSV + Textura:** Mejor modelo MLP (F1: 0.9137)

**Análisis y Fundamentación:**
Las estadísticas de color puros (RGB y HSV) extraen simplemente medias y varianzas de la luz. Las texturas mediante GLCM (*Gray Level Co-occurrence Matrix*) aportan la matriz matemática de contraste, disimilitud y homogeneidad.

* **RGB + Textura con SVM-RBF:** Fue la combinación más efectiva. El kernel RBF $\exp(-\gamma ||x - y||^2)$ logró proyectar estas características combinadas a un espacio infinito donde son perfectamente separables.

**Conclusiones:**
Las características de color solas confunden fuertemente la escena del cielo de las costas con el cielo de las autopistas. Los mejores modelos fueron aquellos que utilizaron la **fusión de características (Color + Texturas)**, lo cual demuestra empíricamente que la visión computacional tradicional requiere que le digamos a la máquina explícitamente qué atributos geométricos y frecuenciales buscar.


## 6. Reto “Clasificación de perros y gatos”

**Implementación y Resultados (Baseline vs Optimización):**
Se trabajó procesando una muestra de 1200 imágenes (600 por clase) directamente a escala de grises, redimensionándolas a 64x64 píxeles y transformándolas (aplanadas) en vectores numéricos continuos de 4096 características. Se dividieron los datos (70% entrenamiento, 30% prueba) y se evaluaron tres enfoques:

* **SVM por defecto (Baseline):** Exactitud de 0.6111 (61.11%).
* **SVM Optimizado (GridSearch):** Exactitud de 0.5944 (59.44%). Los mejores hiperparámetros encontrados fueron `C=1` y `kernel='rbf'`.
* **Random Forest (Alternativo):** Exactitud de 0.5944 (59.44%) con 200 estimadores.

**Análisis Teórico y Fundamentación Conceptual:**
Al aplanar una imagen, cada píxel se convierte en una característica (Feature).

* **Dimensionalidad y Ruido:** Una exactitud cercana al 60% en un problema binario (donde adivinar al azar da el 50%) indica que los modelos clásicos tienen serias dificultades con estas imágenes. Al aplanar la imagen, se pierde por completo la relación espacial 2D (formas de las orejas, ojos, hocico).
* **El comportamiento de SVM:** Sorprendentemente, el SVM por defecto (Baseline) obtuvo un rendimiento ligeramente superior (61.11%) a la versión optimizada con GridSearch (59.44%). La optimización determinó que el mejor kernel era RBF con $C=1$ (que irónicamente es casi el comportamiento por defecto de la librería Scikit-Learn), pero la validación cruzada ($k=3$) del GridSearchCV seleccionó pesos que resultaron en un ligero *overfitting* sobre el set de prueba en comparación con el modelo entrenado sin la penalización cruzada.
* **Bosques Aleatorios (Random Forest):** A pesar de usar 200 árboles de decisión, su rendimiento fue idéntico al SVM optimizado (59.44%). Los árboles sufren con vectores tan largos (4096 variables) donde no hay características fuertemente discriminantes por sí solas (un solo píxel gris no define si es perro o gato).

**Conclusiones:**
En el dataset *Cats&Dogs* se evidencia el verdadero límite del aprendizaje automático clásico para problemas de visión por computadora puros. Aunque se intentó optimizar el parámetro $C$ y los `kernels` del SVM o usar ensambles como Random Forest, el rendimiento se estancó en torno al 60%. Esto fundamenta teóricamente que, para lograr mejoras verdaderamente significativas en clasificación de imágenes complejas, es obligatorio abandonar el enfoque de aplanado de píxeles y transitar hacia técnicas de extracción de características jerárquicas y espaciales en 2D, como las Redes Neuronales Convolucionales (CNN) del Deep Learning.