# preparacion_datos/ — Fase A: preparación de las imágenes (**siempre en CPU**)

Convierte las imágenes JPG del dataset en tensores listos para la red. **No usa la GPU en ningún punto.**

| Archivo | Contenido |
|---|---|
| `preprocesamiento.py` | Sección **SECUENCIAL** (código original del Lab 3: bucle de 1 hilo) y sección **PARALELA** (el mismo trabajo repartido entre todos los núcleos del CPU con `ProcessPoolExecutor`). Las dos producen un resultado **idéntico bit a bit** (se verifica en `tests/`). |

Qué hace cada imagen (op. 4–6 de `docs/01_analisis_secuencial.md`): abrir el JPG de 200×200 → convertir a RGB →
reducir a 64×64 con Lanczos → dividir entre 255 (rango [0, 1]).

Los manifiestos que dicen *qué* imágenes se usan y a qué conjunto pertenecen (70 % train / 15 % validación /
15 % test) están en `data/split_manifest.csv`; no se vuelven a calcular.

**Quién usa esto:** la versión secuencial la usa `secuencial/`; la paralela la usa `paralelo_cuda/`.
