**CC3069 · Computación Paralela y Distribuida · Universidad del Valle de Guatemala · Semestre II, 2026**

# Proyecto 3 — Mejorando el reconocimiento de ASL con CUDA

Se parte del **Laboratorio 3 de Data Science** (clasificador del alfabeto ASL con CNN, MLP y Random Forest,
dataset [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)) y se propone paralelizarlo:
**preparar las imágenes en CPU multinúcleo** y **entrenar la red en GPU con CUDA**.

## Integrantes:
- Joel Jaquez - 23369
- Kevin Villagran - 23584


## Avance por paso del enunciado

| Paso | Qué es | Dónde | Estado |
|---|---|---|---|
| 1 | Análisis de la solución secuencial existente (2 tablas) | [`docs/01`](docs/01_analisis_secuencial.md) | Completado |
| 2 | Medición del tiempo de ejecución secuencial | [`docs/02`](docs/02_medicion_secuencial.md) · `secuencial/` | Completado |
| 3 | Oportunidades de paralelización | [`docs/03`](docs/03_oportunidades_paralelizacion.md) | Completado |
| 4 | Diseño del algoritmo paralelo (con diagramas) | [`docs/04`](docs/04_diseno_algoritmo_paralelo.md) · `docs/diagramas/` | Completado |
| 5 | Implementación con CUDA Python y mediciones | `paralelo_cuda/` · kernel Numba en `paralelo_cuda/kernel_adam_numba.py` | Completado |
| 6 | Análisis y conclusiones | [`docs/06`](docs/06_analisis_y_conclusiones.md) | Completado |

Notebooks: pasos 1 a 4 en [`notebooks/secuencial_y_diseno.ipynb`](notebooks/secuencial_y_diseno.ipynb); pasos 5 y 6 (código
original y versión CUDA, demostración en vivo, mediciones y análisis) en
[`notebooks/pasos_5_y_6_cuda.ipynb`](notebooks/pasos_5_y_6_cuda.ipynb).

## Resultado de los pasos 5 y 6 (secuencial vs. CUDA, medidos en el mismo equipo: i5-12400 + RTX 4060 Ti)

Tiempos medios de 3 repeticiones, 3 épocas, 17,400 imágenes (detalle y salvedades en [`docs/06`](docs/06_analisis_y_conclusiones.md)):

| Fase | Secuencial (Keras, CPU) | Paralelo (CPU multinúcleo + GPU CUDA) | Speedup |
|---|---:|---:|---:|
| Preparar las imágenes | 22.1 s | 4.0 s | 5.5× |
| Entrenar CNN #2 | 110.9 s | 8.7 s | 12.8× |
| **Total del pipeline** | **168.9 s** | **17.7 s** | **9.55×** |

La evidencia de ejecución (capturas del Administrador de Tareas y `nvidia-smi`) está en [`evidencia/`](evidencia/).

## Resultado del paso 2 (solución secuencial del Lab 3; equipo de medición: i7-10750H)

Código original de Keras en CPU, 3 repeticiones (detalle en [`docs/02`](docs/02_medicion_secuencial.md)):

| Tarea | Tiempo medio | % del total |
|---|---:|---:|
| Entrenar las 3 redes (CNN #1 31.3 s · **CNN #2 192.4 s** · MLP 25.4 s) | 249.1 s | **92.5 %** |
| Preparar las 17,400 imágenes (1 núcleo de 12) | 17.1 s | 6.3 % |
| Evaluar en test | 3.1 s | 1.1 % |
| **Total** | **269.3 ± 5.0 s** | 100 % |

Esto orienta el diseño: el entrenamiento es casi todo el tiempo, y preparar las imágenes usa un solo núcleo.

## Estructura

```text
.
├── secuencial/          Solución ORIGINAL del Lab 3 (Keras, CPU), sin modificar, y su medición.   [Pasos 1-2]
├── docs/                Pasos 1-4 del enunciado y diagramas del diseño.                           [Pasos 1-4]
├── preparacion_datos/   Fase A (CPU): decodificar JPG → 64×64 → normalizar. Versión secuencial y paralela.
├── paralelo_cuda/       Versión NUEVA: la red en la GPU con CUDA (PyTorch).                       [Pasos 5-6]
├── comparacion/         Tablas y gráficos secuencial vs. paralelo.
├── notebooks/           Notebooks de los pasos 1-4 y de los pasos 5-6.
├── results/  logs/      Mediciones, gráficos y logs de uso de CPU/GPU.
├── evidencia/           Capturas de pantalla (Administrador de Tareas y nvidia-smi) de las ejecuciones de pc2.
├── tests/               Comprueban que paralelo == secuencial y que las redes tienen los parámetros del Lab 3.
├── extras/              Opcionales, no requeridos por el enunciado.
├── comun/               Rutas, hiperparámetros del Lab 3 y monitor de recursos (compartido).
└── data/                Manifiestos (partición 70/15/15).
```

Cada carpeta tiene su propio `README.md`.

## Cómo reproducir el paso 2

```powershell
# Entorno con TensorFlow (el código original del Lab 3 usa Keras). El dataset va fuera de OneDrive.
python -m venv C:\venvs\proyecto3_tf
C:\venvs\proyecto3_tf\Scripts\python.exe -m pip install tensorflow pandas scikit-learn pillow psutil nvidia-ml-py
$env:ASL_DATA_ROOT = "C:\datasets\asl"      # carpeta que contiene asl_alphabet_train\
C:\venvs\proyecto3_tf\Scripts\python.exe -u secuencial\medir_secuencial.py --tag pc1 --reps 3 --epochs 3
# Tabla y gráficos (entorno con pandas y matplotlib)
python secuencial\resumir_secuencial.py --tag pc1
```

`--tag` identifica el equipo donde se mide (`pc1`, `pc2`, …); los resultados de cada equipo conviven sin pisarse.

## Decisiones y salvedades

* **Los pasos 1 a 3 trabajan sobre el Lab 3 original** (Keras), que es lo que el enunciado pide medir.
* **Port a PyTorch** (solo para el paso 5): TensorFlow ≥ 2.11 no usa GPU en Windows nativo y el proyecto exige CUDA. Las
  redes se portaron conservando capas, inicialización, hiperparámetros y el conteo exacto de parámetros del Lab 3.
* **Épocas fijas (3)** al medir, para que cada repetición haga el mismo trabajo; no mide la calidad del modelo.
* **Datos y entornos fuera de OneDrive** para que no sincronice ~87,000 imágenes ni miles de archivos de librerías.
