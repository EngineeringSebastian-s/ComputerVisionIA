# GUIA2 - Capa IA con FastAPI

Este proyecto ahora funciona como capa de IA dentro de una arquitectura mayor (frontend + backend + IA).

## Arquitectura

- `app/controllers`: controlador HTTP (FastAPI).
- `app/services`: lógica de negocio y orquestación.
- `app/contracts`: interfaz común para ejercicios.
- `app/schemas`: contratos de request/response.
- `app/output`: imágenes y salidas generadas.

## Endpoint principal

- `POST /api/v1/exercises/execute`

Body base:

```json
{
  "type": "ejercicio1"
}
```

Tipos soportados:
- `ejercicio1`
- `ejercicio2`
- `ejercicio3`
- `ejercicio4`
- `ejercicio5`
- `ejercicio6`

Body con opciones:

```json
{
  "type": "ejercicio5",
  "options": {
    "feature_modes": ["rgb+hsv+texture"],
    "resize": [128, 128]
  }
}
```

La respuesta devuelve:
- `summary`: resultados numéricos/reportes.
- `images`: imágenes generadas con ruta y contenido en base64.

## Ejecutar local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Ejecutar con Docker

```bash
docker compose up --build
```

## Salud

- `GET /health`
