# secuencial/ — la solución ORIGINAL del Laboratorio 3 (la base, sin modificar)

Aquí está lo que se hizo en el laboratorio de Data Science, **tal cual**, y el script que lo mide.
Corresponde a los **pasos 1 y 2** del proyecto.

| Archivo | Contenido |
|---|---|
| `lab3_original/` | Código del Lab 3 **sin cambios**: `data_loading.py`, `preprocessing.py`, `models.py` (Keras: CNN #1, CNN #2, MLP, Random Forest) y `augmentation.py`. |
| `medir_secuencial.py` | Ejecuta ese código y mide el tiempo de cada fase (preprocesar, entrenar, evaluar) con 3 repeticiones. |
| `resumir_secuencial.py` | Convierte lo medido en la tabla y los dos gráficos del paso 2 (`results/secuencial_tabla_<tag>.md`, `results/figures/secuencial_*.png`). Solo usa datos del secuencial. |

> Corre con **TensorFlow/Keras en CPU**: en Windows nativo TensorFlow ya no soporta GPU. Necesita su propio entorno
> (`C:\venvs\proyecto3_tf`), distinto del de PyTorch.

```powershell
$env:ASL_DATA_ROOT = "C:\datasets\asl"
C:\venvs\proyecto3_tf\Scripts\python.exe -u secuencial\medir_secuencial.py --tag <nombre> --reps 3 --epochs 3
```

Resultados: `results/benchmark_<tag>_keras_original.csv` y `logs/resources_<tag>_keras_original.csv`.
Análisis: `docs/01_analisis_secuencial.md` y `docs/02_medicion_secuencial.md`.

| Archivo (evidencia de ejecución) | Contenido |
|---|---|
| `monitor_consola.ps1` | Monitor en consola: uso de cada núcleo lógico, memoria y GPU, cada segundo |
| `capturar_evidencia.py` | Captura de pantalla solo de la ventana del monitor, periódica durante una corrida. Resultado en `results/evidencia/` |
