# Evidencia de ejecución: uso de recursos durante la solución secuencial

Capturas de pantalla del uso de cada núcleo lógico del CPU y de la GPU mientras corre el código original del Laboratorio 3.

| Captura | Momento de la ejecución | Qué muestra |
|---|---|---|
| `secuencial_1_preparar_imagenes_un_nucleo.png` | t = 12.5 s · preparación de las imágenes | **Un solo núcleo cargado** (núcleo 2, 72 %); los otros 11 casi libres. CPU total 13.7 % |
| `secuencial_2_entrenar_todos_los_nucleos.png` | t = 36.3 s · entrenamiento de la CNN #2 | **Los 12 núcleos entre 91 y 97 %**. CPU total 92.9 % |

En ambas la GPU permanece en 8 % y 724 MiB (uso del escritorio): TensorFlow en Windows no la utiliza.

## Cómo se obtuvieron
* **Monitor:** [`secuencial/monitor_consola.ps1`](../../secuencial/monitor_consola.ps1) muestra cada segundo el uso de cada núcleo
  lógico (contadores de rendimiento de Windows, la misma fuente que el Administrador de Tareas), la memoria y la GPU
  (`nvidia-smi`).
* **Captura:** [`secuencial/capturar_evidencia.py`](../../secuencial/capturar_evidencia.py) fotografía solo la ventana del monitor
  cada 2 s durante la corrida; cada archivo se etiqueta con la fase que estaba midiendo el programa.
* **Corrida:** el mismo script de medición (`secuencial/medir_secuencial.py`) con un subconjunto de 9,000 imágenes y 1 época de la
  CNN #2, para no interferir con la medición oficial de tiempos ni agotar la memoria. Sus datos están en
  `corrida_evidencia_tiempos.csv` y `corrida_evidencia_recursos.csv` (uso de cada núcleo cada 0.5 s, con la fase).
  **Estos tiempos no son los del paso 2**, solo acompañan a las capturas.
