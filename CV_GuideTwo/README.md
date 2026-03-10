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
