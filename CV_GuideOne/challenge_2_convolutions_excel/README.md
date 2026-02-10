# Documentación: Reto 2 - Convoluciones y Filtrado de Imágenes

## 1. Análisis de Resultados

Basado en la aplicación de diversos kernels de convolución sobre la imagen original (Logo de Bluetooth), se presentan las siguientes observaciones de procesamiento digital de señales:

### Comparativa de Efectos por Kernel

| Kernel | Tipo de Filtro | Resultado Observado | Intensidad / Precisión |
|--------|----------------|---------------------|------------------------|
| **Sobel X** | Detección de Bordes | Bordes verticales claramente definidos. | Alta; mayor que Prewitt X. |
| **Sobel Y** | Detección de Bordes | Bordes superiores e inferiores bien marcados. | Respuesta suave y estable. |
| **Prewitt X** | Detección de Bordes | Bordes detectados correctamente. | Menor intensidad; sensible al ruido. |
| **Prewitt Y** | Detección de Bordes | Bordes horizontales visibles. | Respuesta uniforme pero menos precisa. |
| **Sharpen** | Realce | Bordes más definidos e incremento del contraste. | Alta en zonas de transición. |
| **Gaussian Blur** | Suavizado | Bordes más difusos y reducción de picos extremos. | Ideal como pre-procesamiento. |

---

## 2. Observaciones Técnicas Detalladas

### 2.1 Eficiencia en la Detección de Gradientes

  
**Sobel vs. Prewitt:** El operador **Sobel** demuestra una superioridad técnica al aplicar un peso mayor al píxel central de la fila o columna (valor de 2 o -2), lo que genera una respuesta más robusta frente al ruido en comparación con el operador **Prewitt**, cuya respuesta es más plana y sensible a imperfecciones.


  
**Complementariedad de Ejes:** Se confirma que la detección completa de la estructura requiere la combinación de los componentes X e Y. Mientras Sobel X resalta la verticalidad, Sobel Y es indispensable para capturar las transiciones horizontales del logo.



### 2.2 Impacto del Kernel de Suavizado (Gaussian Blur)

* **Reducción de Ruido:** A diferencia de los filtros de realce, el kernel Gaussiano actúa como un filtro de paso bajo, eliminando las variaciones bruscas de intensidad. Esto se traduce en una "preparación" del dato, eliminando artefactos que podrían causar falsos positivos en una detección de bordes posterior.



### 2.3 Experimentación con Kernels Propios

  
**Análisis No Ortogonal (Kernel 1):** El diseño del primer kernel personalizado permite resaltar bordes inclinados, siendo útil para analizar orientaciones que los filtros clásicos (X/Y) no capturan con precisión.


  
**Detección de Vértices (Kernel 2):** El segundo kernel experimental se especializa en la detección de esquinas y vértices, generando valores negativos alrededor de la figura para acentuar los puntos de cambio de dirección en la geometría del logo.



---

## 3. Conclusiones Generales

1. **La Importancia del Operador Matemático:** No todos los kernels de detección de bordes son iguales. Sobel es preferible para imágenes con gradientes complejos por su estabilidad, mientras que Prewitt ofrece una alternativa de cálculo más uniforme pero limitada.


2. 
**Transformación del Contraste:** Los filtros de **Sharpen** no crean información nueva, sino que acentúan la diferencia de magnitud en las transiciones, lo que facilita la segmentación visual de la figura principal.


3. **Versatilidad de la Convolución:** El proceso de convolución es la piedra angular del procesamiento de imágenes. Cambiando simplemente una matriz de , el sistema puede pasar de "enfocar" una imagen a "difuminarla" o extraer sus características geométricas esenciales.


4. 
**Respuesta no Convencional:** Los kernels personalizados demuestran que es posible diseñar filtros para necesidades específicas (como inclinaciones o vértices) que los estándares de la industria no cubren de forma nativa.


---

## 4. Evidencia de Implementación

* **Matrices Procesadas:** Aplicación de kernels sobre matriz de 8 bits (0-255).
* **Filtros Aplicados:** Sobel (X/Y), Prewitt (X/Y), Sharpen, Gaussian Blur y 2 Kernels experimentales.
