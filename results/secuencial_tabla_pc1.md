### Tiempos de la solución secuencial (Lab 3 original) — pc1

| Tarea | Rep. 1 | Rep. 2 | Rep. 3 | **Media ± desv. (s)** | ms por imagen |
|---|---:|---:|---:|---:|---:|
| Preparar las imágenes | 16.84 | 17.05 | 17.36 | **17.09 ± 0.26** | 0.98 |
| Entrenar CNN #1 | 29.83 | 31.97 | 32.03 | **31.27 ± 1.25** | 0.86 |
| Entrenar CNN #2 | 188.60 | 193.28 | 195.34 | **192.41 ± 3.45** | 5.27 |
| Entrenar MLP | 25.50 | 25.14 | 25.67 | **25.44 ± 0.27** | 0.70 |
| Evaluar en test CNN #1 | 0.54 | 0.58 | 0.59 | **0.57 ± 0.03** | 0.22 |
| Evaluar en test CNN #2 | 2.28 | 2.33 | 2.37 | **2.32 ± 0.04** | 0.89 |
| Evaluar en test MLP | 0.19 | 0.20 | 0.19 | **0.20 ± 0.00** | 0.08 |
| **TOTAL (todas las tareas)** | 263.78 | 270.54 | 273.55 | **269.29 ± 5.00** | |

### Reparto del tiempo

| Grupo | Tiempo medio | % del total |
|---|---:|---:|
| Entrenar | 249.12 s | 92.5 % |
| Preparar las imágenes | 17.09 s | 6.3 % |
| Evaluar en test | 3.09 s | 1.1 % |

_«ms por imagen» = tiempo medio ÷ imágenes procesadas (preparar: 17,400; entrenar: 12,180 × 3 épocas, e incluye la validación de cada época; evaluar: 2,610)._
