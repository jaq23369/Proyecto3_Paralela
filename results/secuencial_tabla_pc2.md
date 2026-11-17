### Tiempos de la solución secuencial (Lab 3 original) — pc2

| Tarea | Rep. 1 | Rep. 2 | Rep. 3 | **Media ± desv. (s)** | ms por imagen |
|---|---:|---:|---:|---:|---:|
| Preparar las imágenes | 36.17 | 15.20 | 14.88 | **22.08 ± 12.20** | 1.27 |
| Entrenar CNN #1 | 19.37 | 21.34 | 21.37 | **20.70 ± 1.14** | 0.57 |
| Entrenar CNN #2 | 109.66 | 111.07 | 112.03 | **110.92 ± 1.19** | 3.04 |
| Entrenar MLP | 12.93 | 13.51 | 12.91 | **13.11 ± 0.34** | 0.36 |
| Evaluar en test CNN #1 | 0.38 | 0.41 | 0.43 | **0.41 ± 0.02** | 0.16 |
| Evaluar en test CNN #2 | 1.54 | 1.59 | 1.51 | **1.55 ± 0.04** | 0.59 |
| Evaluar en test MLP | 0.16 | 0.17 | 0.17 | **0.17 ± 0.00** | 0.06 |
| **TOTAL (todas las tareas)** | 180.22 | 163.29 | 163.31 | **168.94 ± 9.77** | |

### Reparto del tiempo

| Grupo | Tiempo medio | % del total |
|---|---:|---:|
| Entrenar | 144.73 s | 85.7 % |
| Preparar las imágenes | 22.08 s | 13.1 % |
| Evaluar en test | 2.12 s | 1.3 % |

_«ms por imagen» = tiempo medio ÷ imágenes procesadas (preparar: 17,400; entrenar: 12,180 × 3 épocas, e incluye la validación de cada época; evaluar: 2,610)._
