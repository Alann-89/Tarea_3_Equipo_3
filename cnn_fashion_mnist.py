"""
Tarea 3 - Inteligencia Artificial en el Procesamiento de Imágenes
Clasificación de Fashion-MNIST con una CNN sencilla (TensorFlow / Keras).

Uso:     python cnn_fashion_mnist.py
Genera:  figuras/*.png              (figuras del reporte)
         resultados/arquitectura.txt (resumen de Keras)
         resultados/metricas.json    (todas las cifras del reporte)
         resultados/modelo.keras     (modelo entrenado, para reutilizarlo sin reentrenar)
"""
import os, gzip, json
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # backend sin ventana: solo guarda imágenes en disco
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------- 0) REPRODUCIBILIDAD
# set_random_seed fija a la vez las semillas de Python, NumPy y TensorFlow.
# enable_op_determinism obliga a TF a usar operaciones deterministas, de modo que dos
# ejecuciones en la MISMA máquina den exactamente los mismos números (en otra máquina
# o en GPU pueden variar unas décimas por diferencias de hardware).
SEED = 42
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

CLASES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
          "Sandalia", "Camisa", "Tenis", "Bolso", "Botín"]   # orden oficial de etiquetas 0-9
os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

# ---------------------------------------------------------------- 1) DATOS
def leer_idx(ruta, imagenes):
    """Lee un archivo IDX comprimido (.gz) de Fashion-MNIST.
    El encabezado mide 16 bytes en archivos de imágenes y 8 en los de etiquetas."""
    with gzip.open(ruta, "rb") as f:
        datos = np.frombuffer(f.read(), np.uint8, offset=16 if imagenes else 8)
    return datos.reshape(-1, 28, 28) if imagenes else datos

def cargar_datos():
    """Usa las copias locales de data/ si existen; si no, Keras descarga el dataset."""
    if os.path.exists("data/train-images-idx3-ubyte.gz"):
        xtr = leer_idx("data/train-images-idx3-ubyte.gz", True)
        ytr = leer_idx("data/train-labels-idx1-ubyte.gz", False)
        xte = leer_idx("data/t10k-images-idx3-ubyte.gz", True)
        yte = leer_idx("data/t10k-labels-idx1-ubyte.gz", False)
        return (xtr, ytr), (xte, yte)
    return keras.datasets.fashion_mnist.load_data()

(x_train, y_train), (x_test, y_test) = cargar_datos()
print("Entrenamiento:", x_train.shape, " Prueba:", x_test.shape)

# Normalización a [0, 1]: los píxeles vienen en 0-255; valores pequeños y en la misma
# escala hacen que el descenso del gradiente sea más estable.
# [..., None] añade el eje de canal: Conv2D espera (N, alto, ancho, canales) = (N, 28, 28, 1).
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test  = (x_test.astype("float32") / 255.0)[..., None]

# Separación explícita entrenamiento / validación (90 % / 10 %).
# Se toma el último 10 %, igual que haría validation_split=0.1, pero así queda claro
# qué imágenes se usaron para ajustar pesos (54,000) y cuáles solo para validar (6,000).
N_VAL = len(x_train) // 10
x_tr, y_tr   = x_train[:-N_VAL], y_train[:-N_VAL]
x_val, y_val = x_train[-N_VAL:], y_train[-N_VAL:]

# Figura 1: un ejemplo de cada clase
fig, ax = plt.subplots(2, 5, figsize=(9, 5))
for i, a in enumerate(ax.ravel()):
    idx = np.where(y_train == i)[0][0]                # primera imagen de la clase i
    a.imshow(x_train[idx, :, :, 0], cmap="gray"); a.set_title(CLASES[i]); a.axis("off")
plt.suptitle("Fashion-MNIST: un ejemplo por clase"); plt.tight_layout()
plt.savefig("figuras/01_muestras_dataset.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 2) MODELO
# Dos bloques convolución + max-pooling (extractor de características) y un
# clasificador denso. Tamaño de salida de una convolución sin relleno: n - k + 1.
modelo = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation="relu"),   # 28-3+1 = 26 -> 26x26x32; bordes y texturas
    keras.layers.MaxPooling2D(2),                    # 26/2 = 13 -> 13x13x32
    keras.layers.Conv2D(64, 3, activation="relu"),   # 13-3+1 = 11 -> 11x11x64; patrones más complejos
    keras.layers.MaxPooling2D(2),                    # floor(11/2) = 5 -> 5x5x64
    keras.layers.Flatten(),                          # 5*5*64 = 1,600 valores
    keras.layers.Dropout(0.3),                       # apaga 30 % de las entradas al azar (regularización)
    keras.layers.Dense(128, activation="relu"),      # 1,600*128 + 128 = 204,928 parámetros
    keras.layers.Dense(10, activation="softmax"),    # una probabilidad por clase (suman 1)
], name="CNN_sencilla")

# Adam con tasa 1e-3 (valor por defecto recomendado por sus autores).
# "sparse_" porque las etiquetas son enteros 0-9 y no vectores one-hot.
modelo.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

lineas = []
modelo.summary(print_fn=lambda s, **kw: lineas.append(s))
with open("resultados/arquitectura.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lineas))
print("\n".join(lineas))
n_params = modelo.count_params()

# ---------------------------------------------------------------- 3) ENTRENAMIENTO
hist = modelo.fit(x_tr, y_tr, epochs=12, batch_size=128,
                  validation_data=(x_val, y_val), verbose=2)

# Se vuelve a evaluar con el modelo final (dropout desactivado en inferencia).
# La exactitud de entrenamiento se mide solo sobre las 54,000 imágenes usadas para ajustar pesos.
loss_tr, acc_tr   = modelo.evaluate(x_tr, y_tr, verbose=0)
loss_val, acc_val = modelo.evaluate(x_val, y_val, verbose=0)
loss_te, acc_te   = modelo.evaluate(x_test, y_test, verbose=0)
print(f"Exactitud entrenamiento: {acc_tr:.4f} | validación: {acc_val:.4f} | prueba: {acc_te:.4f}")
modelo.save("resultados/modelo.keras")

# Figura 2: curvas de aprendizaje (épocas numeradas desde 1)
h = hist.history
ep = np.arange(1, len(h["loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ep, h["accuracy"], "o-", ms=3, label="Entrenamiento")
ax[0].plot(ep, h["val_accuracy"], "o-", ms=3, label="Validación")
ax[0].set_title("Exactitud (accuracy)"); ax[0].set_xlabel("Época"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, h["loss"], "o-", ms=3, label="Entrenamiento")
ax[1].plot(ep, h["val_loss"], "o-", ms=3, label="Validación")
ax[1].set_title("Pérdida (entropía cruzada)"); ax[1].set_xlabel("Época"); ax[1].legend(); ax[1].grid(alpha=.3)
for a in ax: a.set_xticks(ep)
plt.tight_layout(); plt.savefig("figuras/02_curvas_entrenamiento.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 4) EVALUACIÓN
probs = modelo.predict(x_test, verbose=0)       # (10000, 10) probabilidades
y_pred = probs.argmax(axis=1)                   # clase más probable
cm = confusion_matrix(y_test, y_pred)           # filas: clase real, columnas: predicha

# Figura 3: matriz de confusión
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(CLASES, rotation=45, ha="right"); ax.set_yticklabels(CLASES)
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real"); ax.set_title("Matriz de confusión (prueba)")
plt.tight_layout(); plt.savefig("figuras/03_matriz_confusion.png", dpi=150); plt.close()

reporte = classification_report(y_test, y_pred, target_names=CLASES, digits=4, output_dict=True)
print(classification_report(y_test, y_pred, target_names=CLASES, digits=4))

# Figura 4: los primeros 5 ejemplos de prueba bien clasificados y los primeros 5 mal clasificados
ok  = np.where(y_pred == y_test)[0][:5]
bad = np.where(y_pred != y_test)[0][:5]
fig, ax = plt.subplots(2, 5, figsize=(11, 5.4))
for fila, (indices, color) in enumerate([(ok, "green"), (bad, "red")]):
    for k, idx in enumerate(indices):
        ax[fila, k].imshow(x_test[idx, :, :, 0], cmap="gray"); ax[fila, k].axis("off")
        ax[fila, k].set_title(f"Real: {CLASES[y_test[idx]]}\nPred: {CLASES[y_pred[idx]]}\n"
                              f"({probs[idx].max():.0%})", fontsize=10, color=color)
fig.text(0.01, 0.75, "Correctos", rotation=90, fontsize=12, va="center")
fig.text(0.01, 0.27, "Incorrectos", rotation=90, fontsize=12, va="center")
plt.tight_layout(rect=(0.03, 0, 1, 1)); plt.savefig("figuras/04_ejemplos_clasificacion.png", dpi=150); plt.close()

# Pares (real -> predicha) más confundidos: se ignora la diagonal (aciertos)
cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
pares = sorted([(int(cm_off[i, j]), CLASES[i], CLASES[j]) for i in range(10) for j in range(10) if i != j],
               reverse=True)[:5]

def ejemplo(idx):
    return {"indice": int(idx), "real": CLASES[y_test[idx]], "pred": CLASES[y_pred[idx]],
            "confianza": float(probs[idx].max())}

# ---------------------------------------------------------------- 5) GUARDAR MÉTRICAS
# Todas las cifras del reporte salen de este archivo.
with open("resultados/metricas.json", "w", encoding="utf-8") as f:
    json.dump({"tensorflow": tf.__version__, "keras": keras.__version__,
               "n_train": int(len(x_tr)), "n_val": int(len(x_val)), "n_test": int(len(x_test)),
               "parametros": int(n_params),
               "acc_train": float(acc_tr), "acc_val": float(acc_val), "acc_test": float(acc_te),
               "loss_train": float(loss_tr), "loss_val": float(loss_val), "loss_test": float(loss_te),
               "acc_train_ultima_epoca_keras": float(h["accuracy"][-1]),
               "por_clase": {c: reporte[c] for c in CLASES},
               "macro_avg": reporte["macro avg"],
               "matriz_confusion": cm.tolist(), "pares_confundidos": pares,
               "ejemplos_correctos": [ejemplo(i) for i in ok],
               "ejemplos_incorrectos": [ejemplo(i) for i in bad],
               "historial": {k: [float(v) for v in vs] for k, vs in h.items()}},
              f, indent=2, ensure_ascii=False)
print("Listo.")
