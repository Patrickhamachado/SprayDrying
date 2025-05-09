import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# Configuración
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
MODEL_PATH = 'models/warmup_critic.keras'
PLOT_PATH_LOSS = 'img/warmup_critic_loss.png'
PLOT_PATH_MAE = 'img/warmup_critic_mae.png'
EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 256

os.makedirs('models', exist_ok=True)
os.makedirs('img', exist_ok=True)

# Función de recompensa (la misma usada en entrenamiento DDPG)
def compute_reward(state, action):
    return (
        -0.05 * action[1] +         # Bombeo_Low_Pump_P_401
        -0.05 * state[1] +          # P404_High_Pump_Pressure_CV
        -0.05 * state[5] +          # Tower_Input_Air_Fan_Speed_Feedback
        -0.35 * state[7] +          # Torre_Horno_Flujo_Gas
        -0.05 * state[14] +         # F501_Ciclone_01_Speed
        -0.0125 * state[3] +        # Bombeo_Aero_Boost_FT_0371_Kg_h
        -0.0125 * state[4] +        # Bombeo_Aero_Boost_PT_0371_BAR
        1.2 * state[12]             # Torre_PB_Flujo_Schenck
    )

# Cargar datos
df = pd.read_csv(DATASET_PATH)
print(f"Datos cargados: {df.shape[0]} filas")

# Índices
idx_ext = list(range(9, 19))
idx_state = list(range(19, 35))
idx_action = list(range(1, 9))

# Filtrado: solo donde reset == 1 y se puede calcular reward (ignorar última fila)
valid_rows = df['reset'].values[:-1] == 1
states = df.iloc[:-1, idx_state].values[valid_rows]
externals = df.iloc[:-1, idx_ext].values[valid_rows]
actions = df.iloc[:-1, idx_action].values[valid_rows]
next_states = df.iloc[1:, idx_state].values[valid_rows]

# Calcular recompensas reales con s' y a
rewards = np.array([compute_reward(s2, a) for s2, a in zip(next_states, actions)], dtype=np.float32)

# Entradas: [state + external] y action
X1 = np.concatenate([externals, states], axis=1).astype(np.float32)  # shape (N, 26)
X2 = actions.astype(np.float32)
y = rewards.astype(np.float32)

# División train/val
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

# Definición del modelo crítico
def get_critic():
    state_input = layers.Input(shape=(26,))
    action_input = layers.Input(shape=(8,))
    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(256, activation='relu')(concat)
    out = layers.Dense(256, activation='relu')(out)
    outputs = layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs)

critic = get_critic()
critic.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

# Entrenamiento
history = critic.fit(
    [X1_train, X2_train], y_train,
    validation_data=([X1_val, X2_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Guardar modelo
critic.save(MODEL_PATH)
print(f"Crítico guardado en {MODEL_PATH}")

# Gráfica pérdida
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Pérdida (Loss) del Crítico')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH_LOSS)

# Gráfica MAE
plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Error Absoluto Medio (MAE) del Crítico')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH_MAE)

print("Entrenamiento warm-up del crítico completado.")
