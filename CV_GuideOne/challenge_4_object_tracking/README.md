# Documentación: Reto 4 - Detección de Objetos por Color en Tiempo Real

1. Análisis de Resultados 

Basado en las pruebas de ejecución del algoritmo de visión artificial para la identificación de figuras geométricas planas, se presentan las siguientes observaciones generales:

## Rendimiento del Algoritmo

| Parámetro | Comportamiento Observado | Estado |
|-----------|--------------------------|--------|
| **Gama Cromática** | Detección precisa de figuras en color **Rojo**, **Verde** y **Azul**. | Exitoso |
| **Velocidad de Respuesta** | El recuadro de delimitación (Bounding Box) aparece de forma casi instantánea. | Óptimo |
| **Consistencia de Etiquetado** | La clasificación del color en pantalla corresponde fielmente al objeto real. | Exitoso |
| **Estabilidad de Seguimiento** | Seguimiento continuo de la figura mientras se mantiene sobre un fondo neutro. | Estable |

---

## 2. Observaciones Técnicas Detalladas

2.1 Sensibilidad Lumínica y Ruido Ambiental 

* 
**Dependencia de la Iluminación:** Se identificó que la robustez del sistema está directamente ligada a la calidad de la luz ambiental. 


* 
**Impacto de las Sombras:** La presencia de sombras proyectadas sobre la figura altera los valores del espacio de color (probablemente HSV/RGB), lo que provoca que la detección falle o se pierda momentáneamente. 


* 
**Condiciones Críticas:** En escenarios de baja iluminación, el umbral de segmentación puede no ser alcanzado, resultando en una pérdida de seguimiento del objeto. 



2.2 Segmentación sobre Fondo Neutro 

* 
**Contraste de Fondo:** El uso de una base blanca facilita la binarización de la imagen, permitiendo que los contornos de las figuras roja, verde y azul se aíslen con mayor claridad del resto de la escena. 



---

## 3. Conclusiones Generales

1. 
**Eficacia del Modelo de Color:** El sistema demuestra una alta fiabilidad en la detección de colores primarios bajo condiciones controladas, validando la lógica de filtrado implementada. 


2. 
**Limitaciones de Visión Tradicional:** Al depender de umbrales fijos o rangos de color específicos, el algoritmo es vulnerable a cambios dinámicos en el entorno (luces/sombras), una característica común en sistemas de visión artificial que no utilizan redes neuronales compensatorias. 


3. 
**Optimización de Procesamiento:** La rapidez con la que aparece el recuadro sugiere un procesamiento eficiente de los frames, lo que permite aplicaciones de seguimiento en tiempo real con baja latencia. 
