# Numerical Linear Algebra From Scratch

Implementación educativa de algoritmos clásicos de Álgebra Lineal Numérica, desarrollados en Python y validados con tests automáticos.

## Objetivo

Este repositorio reúne implementaciones manuales (sin depender de `numpy.linalg` para la lógica central) de métodos de:

- operaciones matriciales básicas
- factorización LU/LDV
- factorización QR (Gram-Schmidt y Householder)
- método de la potencia y diagonalización
- cadenas de Markov y matrices ralas
- SVD reducida

Está pensado como base académica y técnica para demostrar fundamentos numéricos, trazabilidad y buenas prácticas de estructura de proyecto.

## Estructura del proyecto

```text
.
├── src/nla/                 # Módulo principal (incluye alc.py)
├── labos/                   # Implementaciones por laboratorio (labo_01...labo_08)
├── tests/                   # Suite de tests ejecutables con python -m
├── notebooks/               # Notebooks de experimentación
├── examples/                # Ejemplos de uso y scripts auxiliares
│   └── images/              # Imágenes usadas por ejemplos
├── data/
│   ├── samples/             # Imágenes simples de referencia
│   ├── weights/W/           # Matrices precomputadas (.npy)
│   └── datasets/            # Dataset cats_and_dogs
└── lab_imports.py               # Atajo de imports para laboratorio/tests
```

## Requisitos

- Python 3.10+
- pip

Dependencias principales:

- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `torch` / `torchvision` (solo para ejemplo de embeddings)
- `requests`

## Instalación rápida

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Ejecución de tests

Desde la raíz del proyecto:

```bash
python -m tests.test_01_fundamentos
python -m tests.test_02_transformaciones
python -m tests.test_03_normas
python -m tests.test_04_factorizacion_lu
python -m tests.test_05_factorizacion_qr
python -m tests.test_06_metodo_potencia
python -m tests.test_07_markov_ralas
python -m tests.test_08_svd
```

## Uso del módulo principal

Compatibilidad actual:

```python
import alc
```

El código fuente principal vive en:

- `src/nla/alc.py`

## Ejemplo de embeddings de imágenes

```bash
python examples/generate_image_embeddings.py
```

El script utiliza imágenes locales en `examples/images/`.

## Estado del proyecto

Proyecto académico consolidado y reorganizado para presentación técnica. Incluye separación clara entre código, datos, ejemplos y notebooks.
