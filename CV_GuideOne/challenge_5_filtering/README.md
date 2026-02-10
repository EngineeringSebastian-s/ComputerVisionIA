# Documentación: Reto 5 - Convoluciones para Filtrado

## 1. Análisis de Resultados

Los tiempos de ejecución obtenidos revelan una disparidad masiva entre el procesamiento manual y el uso de librerías especializadas:

| Método | Tiempo Promedio (s) | Eficiencia Relativa |
| --- | --- | --- |
| **Manual (Python + NumPy)** | **10.8153 s** | 1x (Referencia) |
| **OpenCV (`medianBlur`)** | **0.00045 s** | **~24,000 veces más rápido** |

### Observaciones sobre el Ruido Salt & Pepper

* **Naturaleza del Ruido:** Este ruido impulsivo se manifiesta como píxeles con valores extremos (0 para pimienta, 255 para sal).
* **Eficacia del Filtro:** El filtro de mediana demostró ser la herramienta ideal. A diferencia de un filtro de promedio (como el Gaussiano), la mediana no promedia el ruido con los píxeles vecinos, sino que lo reemplaza por un valor real de la vecindad, eliminando los puntos blancos y negros casi por completo sin emborronar excesivamente los bordes.

---

## 2. Conclusiones y Análisis Técnico

### 2.1 La Superioridad de OpenCV

La abismal diferencia de velocidad (de 10 segundos a menos de un milisegundo) se debe a varios factores críticos:

* **Especialización:** OpenCV no opera con bucles `for` de Python; utiliza implementaciones en **C++ altamente optimizadas** para la arquitectura de la CPU (instrucciones SSE, AVX).
* **Manejo de Matrices:** Mientras que el código manual recorre la matriz píxel por píxel (generando un alto *overhead* en el intérprete de Python), OpenCV trata la imagen como un bloque de memoria contiguo y realiza operaciones vectorizadas.

### 2.2 Convolución Manual vs. Automatizada

* **Lógica de Convolución:** La implementación manual permitió validar el concepto de "ventana deslizante". Se extrae una submatriz de , se calculan los estadísticos y se asigna el resultado al píxel central.
* **Tratamiento de Bordes:** En la versión manual, se ignoraron los bordes (offset), mientras que OpenCV aplica técnicas de *padding* (relleno de bordes) para mantener el tamaño original de la imagen con mayor precisión.

### 2.3 Impacto en la Visión Artificial

* **Preprocesamiento:** Este reto demuestra que la limpieza de ruido es una etapa obligatoria pero costosa. En aplicaciones de tiempo real (como seguimiento de objetos), el uso de filtros manuales sería inviable.
* **Manejo de Datos:** Se confirma que OpenCV no es solo una librería de utilidades, sino un motor de procesamiento de matrices especializado que optimiza el uso del hardware para tareas de visión.

---

## 3. Evidencia de Implementación

El sistema generó y almacenó los siguientes archivos en la ruta local como evidencia del proceso:

1. **`original-gray.jpg`**: Imagen base en escala de grises.
2. **`ruido-salt-pepper.jpg`**: Imagen contaminada con ruido impulsivo al 2%.
3. **`filtro-manual.jpg`**: Resultado tras 10.8 segundos de procesamiento nativo.
4. **`filtro-opencv.jpg`**: Resultado instantáneo usando `cv2.medianBlur`.