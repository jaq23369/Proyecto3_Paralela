# paralelo_cuda/ — versión PARALELA: entrenamiento en GPU con CUDA

Corresponde a los **pasos 5 y 6**. El diseño que implementa está en `docs/03_oportunidades_paralelizacion.md` y
`docs/04_diseno_algoritmo_paralelo.md`. Aquí **solo vive la matemática de la red neuronal**; la preparación de las
imágenes está en `preparacion_datos/` (CPU).

| Archivo | Contenido |
|---|---|
| `modelos.py` | CNN #1, CNN #2 y MLP portadas de Keras a **PyTorch** (mismas capas y mismo número de parámetros que el Lab 3) y el Random Forest (que se queda en CPU). |
| `entrenamiento.py` | Bucle de entrenamiento y evaluación preparado para **CUDA**: datos residentes en la GPU, sin `.item()` por lote, `torch.cuda.synchronize()` antes de medir. |
| `medir_cuda.py` | Mide las fases con CUDA (preprocesar en CPU multinúcleo → copiar a la GPU → entrenar → evaluar). |
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
Código funcional y cubierto por las pruebas de `tests/`. **Pendiente:** ejecutar las mediciones con CUDA (no hay
resultados de esta versión en el repositorio), implementar el kernel explícito de «CUDA Python» y redactar el análisis.
