# SmartPot Lettuce AI

Base profesional para el proyecto de visión por computadora orientado a detectar el estado de salud de cultivos de lechuga hidropónica dentro del ecosistema SmartPot.

El repositorio está preparado para:

- Analizar el dataset de `Lettuce Diseases` de Kaggle.
- Entrenar un clasificador de imágenes con `MobileNetV2` y transfer learning.
- Evaluar el modelo con matriz de confusión, precisión, recall, F1-score y exactitud.
- Exportar artefactos listos para una futura integración con SmartPot.

## Requisitos

- Python 3.14
- `uv`

## Instalación

```bash
uv sync
```

Para usar entrenamiento/evaluación/inferencia con TensorFlow, instala el extra ML en un entorno compatible con TensorFlow:

```bash
uv sync --extra ml
```

Nota importante: en este momento TensorFlow estable no publica wheels para Python 3.14 en todos los casos, así que si vas a entrenar con el stack completo, lo más seguro es usar Python 3.13 o un entorno compatible con los wheels oficiales de TensorFlow.

Si quieres crear el entorno desde cero:

```bash
uv init
uv add tensorflow numpy pandas matplotlib scikit-learn pillow
uv add --dev pytest ruff mypy
```

## Estructura

```text
.
├── data/
│   ├── processed/
│   ├── raw/
│   │   ├── Bacterial/
│   │   ├── Downy_mildew_on_lettuce/
│   │   ├── Healthy/
│   │   ├── Powdery_mildew_on_lettuce/
│   │   ├── Septoria_blight_on_lettuce/
│   │   ├── Shepherd_purse_weeds/
│   │   ├── Viral/
│   │   └── Wilt_and_leaf_blight_on_lettuce/
│   ├── splits/
│   └── test/
├── models/
│   ├── checkpoints/
│   │   └── lettuce_mobilenetv2_best.keras
│   └── exported/
│       ├── class_names.json
│       └── lettuce_mobilenetv2.keras
├── reports/
│   ├── figures/
│   │   └── confusion_matrix.png
│   └── metrics/
│       ├── classification_report.json
│       ├── evaluation_metrics.json
│       └── training_history.json
├── scripts/
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
├── src/
│   └── smartpot_lettuce_ai/
│       ├── cli.py
│       ├── config.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── model.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── utils.py
│       └── __init__.py
├── tests/
│   ├── test_config.py
│   └── test_dataset.py
├── .env.example
├── lettuce-diseases-metadata.json
├── main.py
├── pyproject.toml
└── README.md
```

## Uso rápido

Inspeccionar el metadata del dataset:

```bash
uv run python main.py
```

Entrenar el modelo cuando tengas el dataset organizado por carpetas de clase:

```bash
uv run python scripts/train.py --data-dir data/raw
```

Evaluar un modelo exportado:

```bash
uv run python scripts/evaluate.py --model-path models/exported/lettuce_mobilenetv2.keras --test-dir data/splits/test
```

Predecir una imagen:

```bash
uv run python scripts/predict.py --model-path models/exported/lettuce_mobilenetv2.keras --image path/to/image.jpg
```

## Dataset

El archivo `lettuce-diseases-metadata.json` incluye el metadata del dataset de Kaggle y sirve como referencia para extraer las clases y documentar la procedencia de los datos.

## Notas

- El proyecto está pensado como base modular, no como notebook monolítico.
- La lógica importante vive en `src/smartpot_lettuce_ai/`.
- Los artefactos generados por entrenamiento y evaluación quedan fuera del control de versiones.
