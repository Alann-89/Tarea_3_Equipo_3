# Tarea 3 - IA en el Procesamiento de Imágenes (CNN en Fashion-MNIST)

Procesamiento de Imágenes · Prof. Hermilo Sánchez Cruz · 7º semestre, grupo A · UAA

Equipo: Alan Josué García Quintero, Dairon Tadeo García Medina, Eduardo Isai López García

CNN sencilla (TensorFlow/Keras) para clasificar Fashion-MNIST.

## Ejecución
```bash
pip install -r requirements.txt
python cnn_fashion_mnist.py        # entrena, evalúa y genera figuras/ y resultados/
```
El script usa `data/*.gz` si existen; si no, descarga el dataset con `keras.datasets`.
Los archivos `.gz` se pueden obtener del repositorio oficial: https://github.com/zalandoresearch/fashion-mnist (carpeta `data/fashion`).

## Contenido
- `cnn_fashion_mnist.py`: código comentado (datos, modelo, entrenamiento, evaluación, figuras)
- `figuras/`: muestras, curvas, matriz de confusión, ejemplos correctos/incorrectos
- `resultados/`: `arquitectura.txt`, `metricas.json`, `modelo.keras` (modelo entrenado)
- `Reporte_Tarea3_IA_Procesamiento_Imagenes.pdf`: reporte final

## Resultados (semilla 42, TensorFlow 2.21 en CPU)
| Parámetros | Entrenamiento (54,000) | Validación (6,000) | Prueba (10,000) |
|---|---|---|---|
| 225,034 | 94.48 % | 92.08 % | 91.17 % |

La semilla y `enable_op_determinism()` hacen que el resultado se repita exactamente en la misma máquina;
en otro hardware (u otra versión de TensorFlow, o GPU) las cifras pueden variar algunas décimas.
