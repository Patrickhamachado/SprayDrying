
"""
Script de entrenamiento AWR (Advantage-Weighted Regression)
Entrena únicamente la red actor de forma supervisada offline,
ponderando cada acción por su ventaja (advantage).
Utiliza el crítico warmup para calcular Q(s,a).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# ========== CONFIGURACIÓN ==========
CSV_PATH = "data/datos_Normal_a_e_s_7may.csv"
CRITIC_WARMUP_PATH = "models/warmup_critic.keras"
ACTOR_INIT_PATH = "models/warmup_actor.keras"
PLOT_DIR = "img/AWR"
MODEL_OUT_DIR = "models/AWR"
BATCH_SIZE = 512
EPOCHS = 100
ADV_BETA = 0.2  # Factor de temperatura para el softmax de las ventajas
ADV_CLIP_WEIGHT = 20

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ========== CARGA DE DATOS ==========
df = pd.read_csv(CSV_PATH)
df = df[df['reset'] == 1].reset_index(drop=True)  # Filtrar por reset == 1
action_cols = [f'a{i+1}' for i in range(8)]
external_cols = [f'e{i+1}' for i in range(10)]
state_cols = [f's{i+1}' for i in range(16)]

actions = df[action_cols].values.astype(np.float32)
externals = df[external_cols].values.astype(np.float32)
states = df[state_cols].values.astype(np.float32)

states_input = np.concatenate([externals, states], axis=1)

# ========== CARGA DE MODELOS ==========
critic = tf.keras.models.load_model(CRITIC_WARMUP_PATH)
actor = tf.keras.models.load_model(ACTOR_INIT_PATH)

# ========== CÁLCULO DE VENTAJAS ==========
# AWR clásico: advantage = Q(s,a) - baseline
# baseline ≈ media de Q(s,a) o función constante

q_values = critic.predict([states_input, actions], batch_size=1024).flatten()
baseline = np.mean(q_values)
advantages = q_values - baseline

# ========== PESOS ==========
# Normalización tipo softmax controlada por temperatura ADV_BETA
# weights = tf.nn.softmax(advantages / ADV_BETA).numpy()
weights = tf.exp(advantages / ADV_BETA)
weights = tf.minimum(weights, ADV_CLIP_WEIGHT)


# ========== ENTRENAMIENTO DEL ACTOR ==========
actor.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

history = actor.fit(
    states_input,
    actions,
    sample_weight=weights,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2
)

# ========== GUARDAR ==========
actor.save(os.path.join(MODEL_OUT_DIR, "actor_awr.keras"))

# ========== GRÁFICA ==========
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Actor loss")
plt.title("Pérdida de entrenamiento - AWR")
plt.xlabel("Epoch")
plt.ylabel("MSE ponderado")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/train_loss_awr.png")
plt.close()

print("✅ Entrenamiento AWR finalizado. Modelo y gráfica guardados.")
