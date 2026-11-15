# Paso 3 — Identificación de oportunidades de paralelización

Los pasos 1 y 2 muestran que el entrenamiento (operaciones 9 a 15 de la Tabla 1) es el **92.5 %** del tiempo.

## Operaciones que se proponen paralelizar con CUDA y cómo se benefician

| Operación (Tabla 1) | Qué se ejecuta de forma simultánea | Cómo se beneficia de la GPU |
|---|---|---|
| **9. Convolución + ReLU** | Cada valor de salida, que no depende de los demás (8,388,608 en la capa mayor de la CNN #2, con lote de 64) | Millones de productos punto independientes, con mucha aritmética y acceso regular a memoria |
| **10. BatchNorm, Dropout y pooling** | Cada elemento del tensor (BatchNorm: una reducción por canal) | Operaciones simples sobre millones de elementos; aprovecha el gran ancho de banda de memoria de la GPU |
| **11. Capas densas y softmax** | Cada valor de la matriz resultante | Multiplicación de matrices: la operación para la que está diseñada la GPU |
| **12. Función de pérdida** | Una entropía cruzada por imagen del lote (64) y su promedio | Las 64 pérdidas se calculan a la vez |
| **13. Retropropagación** | Misma estructura que la propagación hacia adelante (≈ 2 veces su trabajo) | Duplica el trabajo paralelizable de cada paso |
| **14. Actualización de pesos (Adam)** | Cada parámetro (2.1 a 6.4 millones), sin dependencias entre ellos | Millones de actualizaciones simultáneas |
| **15–16. Evaluación y predicción** | Los mismos cálculos de la propagación hacia adelante; lotes independientes | Reutiliza los kernels anteriores con los datos ya en la memoria de la GPU |

**Idea clave:** el paralelismo está *dentro* de cada paso (entre los valores de cada capa), no *entre* pasos: cada
paso de entrenamiento necesita los pesos que dejó el anterior, así que los 191 lotes de una época no pueden
ejecutarse a la vez.

## Operaciones que no se llevan a CUDA

| Operación (Tabla 1) | Dónde se ejecuta | Justificación |
|---|---|---|
| **2–7. Preparar las imágenes** | CPU, repartido entre sus núcleos (cada imagen es independiente) | La GPU no abre archivos; decodificar JPEG tiene muchas bifurcaciones y poca aritmética; se hace una sola vez y pesa el 6.3 % del tiempo |
| 1, control de épocas y parada temprana | CPU | Se ejecutan una vez o son decisiones secuenciales |
| 17. Random Forest | CPU | No es una red neuronal; sus árboles ya usan todos los núcleos |
