# extras/ — opcionales, NO requeridos por el enunciado

Scripts de apoyo para los pasos 5 y 6. El proyecto se puede entregar sin ellos. **Ninguno se ha ejecutado en el
estado actual del repositorio.**

| Script | Para qué sirve |
|---|---|
| `compare_cpu_parallelism.py` | Escalamiento del preprocesamiento paralelo en CPU (procesos vs. hilos, 1–12 workers) |
| `profile_kernels.py` | Lista los kernels CUDA reales con su grid y bloque, para contrastarlos con el diseño del paso 4 |
| `train_full.py` | Entrenamiento completo con parada temprana, para comprobar que la GPU no cambia la calidad del modelo (≈ 98 % de la CNN #2 en el Lab 3) |
