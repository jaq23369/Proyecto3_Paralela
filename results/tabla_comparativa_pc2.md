### Tiempos medios ± desviación estándar (3 repeticiones) — pc2

| Fase | Modelo | Secuencial (Lab 3 original) (s) | Paralelo (CPU multinúcleo + GPU CUDA) (s) | **Speedup** |
|---|---|---:|---:|---:|
| Preparar las imágenes (CPU) | — | 22.08 ± 12.20 | 3.99 ± 0.43 | 5.53× |
| Entrenar las redes | CNN #1 | 20.70 ± 1.14 | 3.34 ± 1.73 | 6.20× |
| Entrenar las redes | CNN #2 | 110.92 ± 1.19 | 8.67 ± 2.66 | 12.80× |
| Entrenar las redes | MLP | 13.11 ± 0.34 | 1.15 ± 0.03 | 11.40× |
| Evaluar en test | CNN #1 | 0.41 ± 0.02 | 0.06 ± 0.00 | 6.47× |
| Evaluar en test | CNN #2 | 1.55 ± 0.04 | 0.18 ± 0.00 | 8.43× |
| Evaluar en test | MLP | 0.17 ± 0.00 | 0.00 ± 0.00 | 41.19× |
| **TOTAL del pipeline** | | **168.94 ± 9.77** | **17.70 ± 4.00** | **9.55×** |
| Copiar los datos a la GPU | — | — | 0.30 ± 0.05 | (solo existe en paralelo) |

### Uso de recursos durante cada fase (promedio de los logs de `psutil` y NVML)

| Versión | Fase | CPU (promedio) | Núcleos con > 50 % de uso | GPU (promedio) | VRAM máxima* |
|---|---|---:|---:|---:|---:|
| Secuencial (Lab 3 original) | Preparar las imágenes (CPU) | 20 % | 0 de 12 | 4 % | 1644 MiB |
| Secuencial (Lab 3 original) | Entrenar las redes | 91 % | 12 de 12 | 5 % | 1736 MiB |
| Paralelo (CPU multinúcleo + GPU CUDA) | Preparar las imágenes (CPU) | 84 % | 12 de 12 | 2 % | 2085 MiB |
| Paralelo (CPU multinúcleo + GPU CUDA) | Entrenar las redes | 18 % | 0 de 12 | 76 % | 4447 MiB |

\*La VRAM incluye ≈ 1 GB que ya usa Windows. La inferencia en GPU dura menos que el intervalo de muestreo (0.5 s) y por eso no aparece.

_Inicialización única de CUDA (contexto + autotune de cuDNN), medida aparte y **no** incluida en los tiempos: 5.0 s._
