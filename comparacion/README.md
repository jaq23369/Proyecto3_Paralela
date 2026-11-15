# comparacion/ — secuencial vs. paralelo (para el informe)

`comparar.py` lee las mediciones de `secuencial/` y de `paralelo_cuda/` y genera las tablas y los gráficos del informe.

```powershell
C:\venvs\proyecto3\Scripts\python.exe comparacion\comparar.py --tag <nombre>
```

Requiere las mediciones de ambos lados con el mismo `--tag` (mismo equipo). Salida en `results/`:
`tabla_comparativa_<tag>.md`, `speedup_<tag>.csv` y `figures/1_tiempo_total_…`, `2_speedup_por_fase_…`, `3_uso_recursos_…`.
