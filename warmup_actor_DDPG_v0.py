import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import random

# Configuración
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
MODEL_PATH = 'models/warmup_actor.keras'
PLOT_PATH_LOSS = 'img/warmup_actor_loss.png'
PLOT_PATH_MAE = 'img/warmup_actor_mae.png'
EPOCHS = 100
LEARNING_RATE = 5e-5
BATCH_SIZE = 256
SEED = 5292

# Crear carpeta si no existe
os.makedirs('models', exist_ok=True)
os.makedirs('img', exist_ok=True)

# Inicializar semilla para reproducibilidad
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# Cargar y preparar los datos
df = pd.read_csv(DATASET_PATH)
print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# Índices
idx_externas = list(range(9, 19))
idx_estados = list(range(19, 35))
idx_entrada = idx_externas + idx_estados
idx_acciones = list(range(1, 9))  # a1-a8

# Filtrar solo donde reset == 1 (ignorar última fila)
valid_rows = df['reset'].values[:-1] == 1
X = df.iloc[:-1, idx_entrada].values[valid_rows]
y = df.iloc[:-1, idx_acciones].values[valid_rows]

# Dividir en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir modelo actor
def get_actor():
    inputs = tf.keras.Input(shape=(26,))
    out = tf.keras.layers.Dense(256, activation='relu')(inputs)
    out = tf.keras.layers.Dense(256, activation='relu')(out)
    outputs = tf.keras.layers.Dense(8, activation='sigmoid')(out)  # Acciones normalizadas [0,1]
    return tf.keras.Model(inputs, outputs)

# Instanciar modelo
actor = get_actor()
actor.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

# Entrenamiento
history = actor.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Guardar modelo
actor.save(MODEL_PATH)
print(f"Actor guardado en {MODEL_PATH}")

# Graficar pérdida (loss)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Pérdida (Loss) del Actor')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH_LOSS)

# Graficar MAE
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Error Absoluto Medio (MAE) del Actor')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH_MAE)

print("Entrenamiento de warm-up del actor completado.")
