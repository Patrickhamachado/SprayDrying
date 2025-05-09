import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os


# Rutas
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
MODEL_PATH = 'models/modelo_torre.keras'
PLOT_PATH = 'img/metricas_entrenamiento_modelo_torre.png'
EPOCAS_FOLD = 80           # Épocas de entrenamiento por cada fold 100
LEARNING_RATE = 1e-3        # Learning rate

# Cargar y preparar los datos
df = pd.read_csv(DATASET_PATH)
print(f"Datos importados, {df.shape[0]} filas, {df.shape[1]} columnas")

# Definir índices
idx_acciones = list(range(1, 9))
idx_externas = list(range(9, 19))
idx_estados = list(range(19, 35))  # s1 a s16
idx_target = list(range(19, 35))   # mismos índices, pero fila siguiente

# Filtrar datos válidos para entrenamiento (por ejemplo, done == False)
valid_rows = df['reset'].values[:-1] == 1  # última fila no tiene target
X = df.iloc[:-1, idx_acciones + idx_externas + idx_estados].values[valid_rows]
y = df.iloc[1:, idx_estados].values[valid_rows]  # siguiente estado

# Arquitectura del modelo
def crear_modelo_torre(input_dim = 34, output_dim = 16):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='sigmoid')
    ])


    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(input_dim,)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     # tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     # tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(output_dim, activation='sigmoid')
    # ])

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss='mse', metrics=['mae'] )

    optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = LEARNING_RATE,
        decay_steps = 1000,
        decay_rate = 0.96
    ))

    model.compile(optimizer = optimizer, loss='mse', metrics=['mae'])

    return model

# Validación cruzada K-Fold
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
fold = 1
history_per_fold = []

# Entrenamiento
for train_idx, val_idx in kf.split(X):
    print(f"\n🔁 Entrenando fold {fold}/{K}...")
    model = crear_modelo_torre()

    history = model.fit(
        X[train_idx], y[train_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs= EPOCAS_FOLD,
        batch_size=256,
        verbose=1
    )

    # Guardar historia
    history_per_fold.append(history.history)

    # Opcional: guardar modelo por fold
    model.save(f"models/modelo_torre_f{fold}.keras")
    fold += 1

# Graficar pérdidas
plt.figure(figsize=(12, 6))
for i, hist in enumerate(history_per_fold):
    plt.plot(hist['val_loss'], label=f'Fold {i+1} val_loss')
plt.title("Pérdida de validación por fold")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
# plt.show()

# Graficar MAE de validación por fold
plt.figure(figsize=(12, 6))
for i, hist in enumerate(history_per_fold):
    if 'val_mae' in hist:
        plt.plot(hist['val_mae'], label=f'Fold {i+1} val_mae')
plt.title("Error Absoluto Medio (MAE) de validación por fold")
plt.xlabel("Épocas")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('img/entrenamiento_modelo_torre_MAE.png')
# plt.show()

# Guardar modelo
model.save(MODEL_PATH)
