# Informe de Proyecto: Aprendizaje Profundo e IA Generativa

**Institución:** Politécnico Colombiano Jaime Isaza Cadavid
**Asignatura:** Visión por computador e Inteligencia Artificial
**Guía Práctica:** 4

---

## 1. Introducción y Contexto de Estudio

El procesamiento avanzado de imágenes mediante técnicas de Aprendizaje Profundo (Deep Learning) e Inteligencia Artificial Generativa ha revolucionado la visión por computador. A diferencia de las metodologías clásicas de aprendizaje automático (Machine Learning), las arquitecturas profundas permiten extraer jerarquías de características visuales complejas directamente de los píxeles. Sin embargo, su desempeño depende drásticamente de la arquitectura elegida, el tamaño del conjunto de datos y la naturaleza del dominio de las imágenes.

Este documento presenta el desarrollo y los resultados de la **Guía Práctica 4**. El proyecto explora y contrasta el rendimiento de arquitecturas secuenciales (entrenadas desde cero) frente a modelos no secuenciales con transferencia de aprendizaje (como ResNet50) en tareas de clasificación. Posteriormente, se evalúa la eficacia de arquitecturas generativas y Transformers (Autoencoders y SwinIR) para la mitigación del ruido multiplicativo *speckle* en imágenes de Radar de Apertura Sintética (SAR).

### Objetivos y Retos del Proyecto

Para abordar el estudio de estas arquitecturas, el informe documenta la solución de los siguientes cuatro retos:

1. **Reto 1 — Clasificación de 3 escenas:** Comparativa entre una CNN secuencial y ResNet50 sobre un dataset de paisajes.
2. **Reto 2 — Clasificación LandUse:** Análisis de rendimiento de modelos profundos sobre texturas de cobertura terrestre.
3. **Reto 3 — Autoencoder para reducción de speckle:** Implementación de una red codificadora-decodificadora para restaurar imágenes SAR.
4. **Reto 4 — Restauración con Transformer (SwinIR):** Evaluación de un modelo generativo preentrenado sobre el dominio de radar.

---

## 2. Reto 1: Clasificación de 3 escenas (coast, forest, highway)

### Procedimiento Implementado

[cite_start]Para la tarea de clasificar imágenes en tres escenas específicas (*coast*, *forest*, *highway*) [cite: 1040][cite_start], en este reto se implementaron dos enfoques: una CNN secuencial entrenada desde cero y una arquitectura no secuencial basada en ResNet50 con transferencia de aprendizaje[cite: 1041].

### Análisis de Resultados

* [cite_start]**CNN Secuencial:** La CNN secuencial presentó problemas de generalización[cite: 1042]. [cite_start]Aunque alcanzó valores cercanos al 99% en entrenamiento, su desempeño en validación fue bajo (alrededor del 52%), evidenciando sobreajuste[cite: 1042]. [cite_start]Además, en algunas ejecuciones mostró colapso hacia una sola clase, lo que indica inestabilidad del modelo[cite: 1043].
* [cite_start]**ResNet50 (Transfer Learning):** En contraste, ResNet50 obtuvo resultados sobresalientes, alcanzando aproximadamente 99.58% de accuracy con imágenes de 224x224[cite: 1044]. [cite_start]Esto demuestra que la transferencia de aprendizaje permite capturar características visuales más complejas y generalizar mejor con datasets limitados[cite: 1045].

### Conclusión del Reto 1

[cite_start]Las arquitecturas no secuenciales preentrenadas son significativamente superiores a las CNN entrenadas desde cero en este tipo de problema[cite: 1046].

---

## 3. Reto 2: Clasificación LandUse (airplane, denseresidential, harbor)

### Procedimiento Implementado

[cite_start]En este reto se aplicó el mismo enfoque sobre el dataset LandUse con tres clases específicas [cite: 1048][cite_start]: *airplane*, *denseresidential* y *harbor*[cite: 1047]. El objetivo fue evaluar si los patrones de comportamiento de los modelos observados en el Reto 1 se mantenían al cambiar el dominio a texturas de cobertura terrestre.

### Análisis de Resultados

* [cite_start]La CNN secuencial volvió a presentar problemas importantes, incluyendo colapso hacia una sola clase (*airplane*), incluso después de aplicar técnicas como reducción del learning rate y uso de pesos de clase[cite: 1049].
* [cite_start]Esto evidencia que el modelo es inestable y sensible al dataset cuando se entrena desde cero[cite: 1050].
* [cite_start]Las causas principales incluyen el tamaño limitado del dataset, la variabilidad entre clases y la ausencia de conocimiento previo en el modelo[cite: 1051].

### Conclusión del Reto 2

[cite_start]La CNN secuencial no es adecuada para este escenario, y nuevamente el uso de transferencia de aprendizaje representa la solución más robusta[cite: 1052].

---

## 4. Reto 3: Autoencoder para reducción de speckle

### Procedimiento Implementado

[cite_start]Se implementó un autoencoder convolucional para reducir ruido speckle en imágenes SAR, utilizando pares de imágenes noisy como entrada y gtruth como referencia[cite: 1053, 1054]. Se calcularon métricas de calidad de imagen para evaluar objetivamente la capacidad de la red para restaurar la señal original.

### Resultados Cuantitativos

| Métrica | Imagen Noisy (Entrada) | Imagen Autoencoder (Salida) | Ground Truth (Referencia) |
| :--- | :--- | :--- | :--- |
| **PSNR** | [cite_start]25.80 [cite: 1056] | [cite_start]22.66 [cite: 1056] | - |
| **SSIM** | [cite_start]0.859 [cite: 1057] | [cite_start]0.824 [cite: 1057] | - |
| **ENL** | [cite_start]0.603 [cite: 1058] | [cite_start]0.594 [cite: 1058] | [cite_start]0.586 [cite: 1058] |

### Análisis de Resultados

* [cite_start]Las métricas muestran que el autoencoder no logró mejorar la calidad de las imágenes[cite: 1059].
* [cite_start]Tanto el PSNR como el SSIM disminuyeron, lo que indica que la imagen reconstruida se alejó del ground truth[cite: 1060].
* [cite_start]El ENL tampoco presentó una mejora significativa[cite: 1061].
* [cite_start]Este comportamiento puede explicarse por una arquitectura limitada, el uso de una función de pérdida no óptima y la posible pérdida de detalles debido al suavizado excesivo[cite: 1062].

### Conclusión del Reto 3

[cite_start]El autoencoder no logró eliminar el ruido speckle de manera efectiva y requiere ajustes en arquitectura, función de pérdida y proceso de entrenamiento[cite: 1063].

---

## 5. Reto 4: Restauración con Transformer (SwinIR)

### Procedimiento Implementado

[cite_start]En este reto se aplicó SwinIR, un modelo basado en Transformer, utilizando imágenes noisy como entrada y comparando contra gtruth[cite: 1064, 1065]. Este enfoque busca aprovechar los mecanismos de autoatención para la reconstrucción de imágenes.

### Resultados Cuantitativos

| Métrica | Imagen Noisy (Entrada) | Imagen SwinIR (Salida) | Ground Truth (Referencia) |
| :--- | :--- | :--- | :--- |
| **PSNR** | [cite_start]19.91 [cite: 1067] | [cite_start]19.64 [cite: 1067] | - |
| **SSIM** | [cite_start]0.623 [cite: 1068] | [cite_start]0.611 [cite: 1068] | - |
| **ENL** | [cite_start]25.57 [cite: 1069] | [cite_start]23.61 [cite: 1069] | [cite_start]91.87 [cite: 1069] |

### Análisis de Resultados

* [cite_start]Los resultados obtenidos evidencian que SwinIR no mejoró la calidad de las imágenes, sino que las degradó ligeramente[cite: 1066, 1070].
* [cite_start]La causa principal es que el modelo fue preentrenado con imágenes naturales y no está adaptado al dominio SAR ni al ruido speckle[cite: 1071].

### Conclusión del Reto 4

[cite_start]Aunque SwinIR cumple con el uso de modelos basados en Transformer, no generaliza adecuadamente a imágenes SAR sin un ajuste específico al dominio[cite: 1072].

---

## 6. Conclusiones Finales del Proyecto

El desarrollo de esta guía práctica ha permitido comprobar empíricamente los desafíos del diseño y entrenamiento de arquitecturas profundas aplicadas a la visión por computador:

* **Superioridad del Transfer Learning en Clasificación:** En dominios donde los datos son limitados o altamente variables (como *3 scenes* o *LandUse*), entrenar redes convolucionales secuenciales desde cero conduce a inestabilidad, colapso de clases y sobreajuste severo. Los modelos con transferencia de aprendizaje (ResNet50) demuestran ser la estrategia óptima y más robusta.
* **El Reto de la Restauración en el Dominio SAR:** Las tareas de reducción de ruido *speckle* resultaron altamente complejas. Ni un autoencoder básico diseñado desde cero, ni un Transformer avanzado (SwinIR) preentrenado en imágenes ópticas, lograron acercarse a la calidad del *Ground Truth*. Esto subraya que los modelos generativos requieren arquitecturas específicamente diseñadas y un reentrenamiento riguroso (fine-tuning) orientado a la naturaleza física del radar de apertura sintética para no degradar las métricas de evaluación objetiva (PSNR, SSIM, ENL).

---

## 7. Estructura del Repositorio e Instrucciones de Ejecución

Este proyecto ha sido modularizado en *scripts* de Python para facilitar la evaluación individual de cada reto.

### Estructura de Archivos Principal

* `reto1.py`: Código para la clasificación de 3 escenas (CNN Secuencial vs ResNet50).
* `reto2.py`: Implementación de la clasificación de cobertura terrestre (LandUse).
* `reto3.py`: Entrenamiento y evaluación del Autoencoder convolucional para el filtrado SAR.
* `reto4.py`: Implementación y evaluación del modelo Transformer SwinIR para la restauración.
* Directorios de Salida: Las subcarpetas generadas por los scripts contendrán los resultados de las métricas (`.csv`), matrices de confusión y parches procesados (`/evidencias_landuse`, `/evidencias_autoencoder`, `/evidencias_swinir`, etc.).