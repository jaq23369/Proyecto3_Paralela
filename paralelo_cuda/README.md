# paralelo_cuda/ — versión PARALELA: entrenamiento en GPU con CUDA

Corresponde a los **pasos 5 y 6**. El diseño que implementa está en `docs/03_oportunidades_paralelizacion.md` y
`docs/04_diseno_algoritmo_paralelo.md`. Aquí **solo vive la matemática de la red neuronal**; la preparación de las
imágenes está en `preparacion_datos/` (CPU).

| Archivo | Contenido |
|---|---|
| `modelos.py` | CNN #1, CNN #2 y MLP portadas de Keras a **PyTorch** (mismas capas y mismo número de parámetros que el Lab 3) y el Random Forest (que se queda en CPU). |
| `entrenamiento.py` | Bucle de entrenamiento y evaluación preparado para **CUDA**: datos residentes en la GPU, sin `.item()` por lote, `torch.cuda.synchronize()` antes de medir. |
| `medir_cuda.py` | Mide las fases con CUDA (preprocesar en CPU multinúcleo → copiar a la GPU → entrenar → evaluar). |
| `kernel_adam_numba.py` | Kernel CUDA propio con **Numba** para el paso de Adam (un hilo por parámetro, bloques de 256). Se valida contra `torch.optim.Adam`, se mide por separado y se puede usar en el entrenamiento con `medir_cuda.py --adam numba` (variante; las mediciones oficiales usan `torch.optim.Adam`). |
| `medicion.py` | Lógica interna de `medir_cuda.py`. Está separada para que los procesos hijos del preprocesamiento no importen PyTorch al arrancar. |

```powershell
$env:ASL_DATA_ROOT = "C:\datasets\asl"
C:\venvs\proyecto3\Scripts\python.exe -u paralelo_cuda\medir_cuda.py --mode cuda --tag <equipo> --reps 3 --epochs 3
C:\venvs\proyecto3\Scripts\python.exe comparacion\comparar.py --tag <equipo>
```

Resultados: `results/benchmark_<equipo>_cuda.csv` y `logs/resources_<equipo>_cuda.csv`.

> `--mode sequential` y `--mode cpu_mt` ejecutan estas mismas redes en CPU (1 hilo / todos los hilos): son referencias
> **opcionales**; la línea base oficial del proyecto es `secuencial/`.

## Estado
Completado: código cubierto por las pruebas de `tests/`, mediciones con CUDA del equipo `pc2`
(`results/benchmark_pc2_cuda.csv`, `logs/resources_pc2_cuda.csv`), kernel de Numba validado
(`results/kernel_adam_numba_pc2.csv`) y análisis en `docs/06_analisis_y_conclusiones.md`.

El kernel de Numba necesita `numba-cuda[cu12]` con NumPy < 2.4, y cargar `nvJitLink` **antes** de `import torch` (el
módulo ya lo hace); por eso se ejecuta como script aparte y no desde `medir_cuda.py`:

```powershell
C:\venvs\proyecto3\Scripts\python.exe paralelo_cuda\kernel_adam_numba.py --tag <equipo> --model cnn2
```
