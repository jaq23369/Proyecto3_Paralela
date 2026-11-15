# Paso 1 — Análisis de la solución secuencial existente

Solución analizada: **Laboratorio 3 de Data Science (Reconocimiento del alfabeto ASL)**, en Keras/TensorFlow.
Código en [`secuencial/lab3_original/`](../secuencial/lab3_original/).

## 1.1 Partes principales del programa y su función

| Parte | Archivo | Función |
|---|---|---|
| Carga y muestreo | `data_loading.py` | Toma 600 imágenes por clase (17,400 de 87,000) |
| Preparación | `preprocessing.py` | Divide en entrenamiento / validación / prueba (70 / 15 / 15 %) y convierte cada imagen en un arreglo 64 × 64 × 3 |
| Modelos | `models.py` | Define CNN #1, CNN #2, MLP y Random Forest |
| Aumento de datos | `augmentation.py` | Transformaciones aleatorias que no cambian la letra |
| Entrenamiento y evaluación | notebooks 04–09 | Entrena con Adam (lote de 64, hasta 30 épocas), valida en cada época y evalúa en prueba |

## 1.2 Descomposición en operaciones

**Tabla 1.** Operaciones, desde la recepción de las imágenes hasta el resultado.

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Muestreo y partición | Elige 600 imágenes por clase y las reparte en entrenamiento (12,180), validación (2,610) y prueba (2,610). Se hace una sola vez |
| 2 | Lectura del archivo | Abre el archivo JPG desde el disco |
| 3 | Decodificación del JPEG | Obtiene la matriz de píxeles de 200 × 200 × 3 |
| 4 | Conversión a RGB | Garantiza tres canales de color |
| 5 | Reducción de resolución | Interpolación Lanczos de 200 × 200 a 64 × 64 |
| 6 | Normalización | Divide los valores entre 255 (rango 0 a 1) |
| 7 | Apilado y etiquetado | Junta las imágenes en un tensor `(N, 64, 64, 3)` y convierte cada letra en un número (0–28) |
| 8 | Aumento de datos *(solo en los reentrenamientos con augmentation)* | Aplica una transformación aleatoria a cada imagen de entrenamiento en cada época |
| 9 | Convolución + ReLU | Filtros de 3 × 3 deslizados sobre la imagen; cada valor de salida es una suma de productos |
| 10 | BatchNorm, Dropout y pooling | Normaliza por canal, apaga neuronas al azar y reduce el mapa a la mitad (máximo de cada 2 × 2) |
| 11 | Capas densas y softmax | Multiplica por matrices de pesos y produce 29 probabilidades |
| 12 | Función de pérdida | Entropía cruzada entre la predicción y la letra real, promediada sobre el lote |
| 13 | Retropropagación | Gradiente de la pérdida respecto a cada peso, de la salida a la entrada |
| 14 | Actualización de pesos | El optimizador Adam corrige cada peso con su gradiente |
| 15 | Evaluación en validación | Al final de cada época calcula el accuracy en validación y decide la parada temprana |
| 16 | Predicción en prueba | Predice las 2,610 imágenes de prueba: accuracy y matriz de confusión (resultado final) |
| 17 | Random Forest *(modelo alterno)* | Entrena 400 árboles sobre los 12,288 píxeles aplanados |

## 1.3 Operaciones repetitivas

Datos: 12,180 imágenes de entrenamiento, lotes de 64 → **191 lotes por época**, hasta 30 épocas por modelo.

**Tabla 2.** Operaciones que se ejecutan repetidamente.

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Leer, decodificar, reducir y normalizar una imagen (ops. 2–6) | Un JPG de 200 × 200 × 3 → arreglo de 64 × 64 × 3 | **17,400** por ejecución (una por imagen, independientes entre sí) |
| Paso de entrenamiento: adelante, pérdida, atrás y Adam (ops. 9–14) | Un lote de 64 imágenes | **191 por época**; hasta 5,730 por modelo (191 × 30) |
| Convolución 3 × 3 (op. 9) | Cada valor de cada mapa de salida: un producto punto de `3 × 3 × canales` valores | Por imagen y por paso. CNN #2: ≈ **118.9 M** multiplicaciones-acumulaciones por imagen (CNN #1: 24.5 M; MLP: 6.4 M) |
| Actualización de pesos Adam (op. 14) | Cada parámetro: 2.1 M (CNN #1), 2.2 M (CNN #2), 6.4 M (MLP) | Una vez por paso de entrenamiento |
| Evaluación en validación (op. 15) | 2,610 imágenes | Una vez por época |
| Predicción en prueba (op. 16) | 2,610 imágenes | Una vez por modelo |
| Aumento de datos (op. 8) | Cada imagen de entrenamiento | 12,180 por época |
