# Documentación: Reto 3 - Análisis Cromatico (ROI)

## 1. Análisis de Resultados por Región

A partir de los logs de ejecución, se observa una correlación directa entre los valores numéricos de los canales *
*Azul (B)**, **Verde (G)** y **Rojo (R)** y la percepción visual del color.

### 1.1 Correspondencia de Medias ()

* **Amarillo (Región 1):** Presenta valores altos en Verde (146.58) y Rojo (188.27), con un Azul muy bajo (10.58). Esto
  confirma la teoría del color donde la mezcla de luz roja y verde produce amarillo.
* **Blanco (Región 4):** Los tres canales tienen valores altos y equilibrados (~215, 216, 202). En el modelo aditivo, la
  saturación similar de los tres canales tiende al blanco.
* **Azul (Región 3):** El canal Azul domina claramente (120.48) sobre el Verde (46.68) y el Rojo (1.67), validando la
  pureza cromática de la zona elegida.

### 1.2 Análisis de la Desviación Estándar ()

* **Homogeneidad (Regiones 1-5):** Las desviaciones estándar son bajas (generalmente < 15). Esto indica que los colores
  son planos y uniformes; existe poca variación entre los píxeles de la muestra.
* **Región Mixta (Región 6):** Tal como se esperaba, la desviación estándar se dispara (44.41, 51.64, 55.37). Esto
  demuestra que la zona contiene una mezcla heterogénea de colores, lo que genera una alta dispersión de datos respecto
  a la media.

---

## 2. Conclusiones Técnicas

1. **Modelo BGR en OpenCV:** Se valida que OpenCV procesa las imágenes en formato **BGR** (Blue-Green-Red). Ignorar este
   orden resultaría en una interpretación errónea de los datos (confundiendo azules con rojos).
2. **Sensibilidad de la Desviación Estándar:** La es un excelente indicador de textura y homogeneidad. Valores altos de
   desviación estándar delatan la presencia de bordes, patrones o ruido en una región de interés.
3. **Identificación Numérica del Color:** Los valores medios obtenidos permiten definir "firmas cromáticas". Estas
   firmas son la base para algoritmos de segmentación por color (como `cv2.inRange`), donde se definen umbrales basados
   en la media la desviación estándar.
4. **Calidad de la Imagen:** Incluso en regiones homogéneas, la desviación estándar no es cero. Esto se debe a la
   compresión de la imagen (artefactos JPEG) y al ruido propio del sensor que capturó la fotografía original.

---

## 3. Evidencia Generada

* **Forma de la imagen:** 1024 px (alto) x 1280 px (ancho) x 3 canales.
* **Imagen Resultante:** `resultado_rois.jpg`. Esta imagen incluye los 6 recuadros dibujados y etiquetados, permitiendo
  la verificación visual de las zonas analizadas estadísticamente en el log de consola.