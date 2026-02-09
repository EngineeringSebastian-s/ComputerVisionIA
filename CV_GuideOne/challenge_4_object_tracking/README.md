## 1. Captura de video

```python
cap = cv2.VideoCapture(0)
```

Se inicializa la captura de video usando la cámara por defecto del sistema.

---

## 2. Definición de rangos de color en HSV

```python
color_ranges = { ... }
```

Se definen los rangos HSV para cada color:

* **Rojo**: requiere dos rangos debido a la discontinuidad del canal Hue.
* **Verde** y **Azul**: usan un único rango.

Estos rangos permiten segmentar únicamente la figura del color deseado.

---

## 3. Conversión de espacio de color

```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

Cada frame capturado se convierte de BGR a HSV para facilitar la segmentación por color.

---

## 4. Segmentación por color

```python
mask = cv2.inRange(hsv, lower, upper)
```

Se genera una máscara binaria donde:

* Blanco → píxeles del color buscado
* Negro → resto de la imagen

Cuando un color tiene más de un rango (rojo), las máscaras se combinan.

---

## 5. Limpieza de la máscara

```python
cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
```

Se aplican operaciones morfológicas para:

* Eliminar ruido
* Rellenar huecos
* Mejorar la forma del objeto detectado

---

## 6. Detección de contornos

```python
contours, _ = cv2.findContours(...)
```

Se detectan los contornos externos de la figura segmentada.

Se filtran por área para evitar falsos positivos pequeños.

---

## 7. Seguimiento y visualización

```python
cv2.rectangle(...)
cv2.putText(...)
```

Para cada figura válida:

* Se dibuja un **recuadro (bounding box)**.
* Se escribe el nombre del color detectado sobre la imagen original.

---

## 8. Visualización en tiempo real

```python
cv2.imshow(...)
```

Se muestra el video procesado en tiempo real.

El programa finaliza al presionar la tecla **ESC**.