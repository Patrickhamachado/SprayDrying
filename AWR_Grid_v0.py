import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random

# ==== CONFIGURACIONES GLOBALES ====
CSV_PATH = "data/datos_Normal_a_e_s_7may.csv"
WARMUP_ACTOR_PATH = "models/warmup_actor.keras"
WARMUP_CRITIC_PATH = "models/warmup_critic.keras"
OUTPUT_DIR = "models/AWR/Grid"

NUM_EPOCHS = 100
BATCH_SIZE = 64
SEED = 5292

# ==== HIPERPARÁMETROS A PROBAR ====
ADV_BETAS = [0.03, 0.05]
ADV_CLIPS = [10.0]

# ==== ESTABLECER SEMILLA GLOBAL ====
def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# ==== CARGAR DATOS ====
df = pd.read_csv(CSV_PATH)
df = df[df['reset'] == 1].reset_index(drop=True)  # Filtrar por reset == 1
state_cols = [f"s{i+1}" for i in range(16)]
external_cols = [f"e{i+1}" for i in range(10)]
action_cols = [f"a{i+1}" for i in range(8)]

states = df[state_cols].values.astype(np.float32)
externals = df[external_cols].values.astype(np.float32)
actions = df[action_cols].values.astype(np.float32)
reset_mask = df["reset"].values.astype(bool)

# Filtrar solo donde reset == 1
states = states[reset_mask]
externals = externals[reset_mask]
actions = actions[reset_mask]

# ==== DEFINIR MODELO ACTOR ====
def get_actor():
    inputs = tf.keras.Input(shape=(26,))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(8, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

# ==== ENTRENAMIENTO AWR ====
def train_awr(beta, clip_weight, save_dir):
    print(f"\n🔧 Entrenando AWR con beta={beta}, clip={clip_weight}")

    set_seed(SEED)

    # Cargar modelos warmup
    actor = tf.keras.models.load_model(WARMUP_ACTOR_PATH)
    critic = tf.keras.models.load_model(WARMUP_CRITIC_PATH)

    # Entradas
    inputs = np.concatenate([externals, states], axis=1)

    # Calcular ventajas
    q_vals = critic.predict([inputs, actions], verbose = 0).squeeze()
    baseline = np.mean(q_vals)
    advantages = q_vals - baseline

    # Ponderaciones
    if clip_weight is not None:
        weights = np.exp(advantages / beta)
        weights = weights / np.max(weights)                 # Normalización
        weights = np.clip(weights, 0, clip_weight)
    else:
        weights = np.exp(advantages / beta)
        weights = weights / np.max(weights)  # Normalización

    weights = weights.astype(np.float32)

    # Compilar modelo
    actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    # Entrenar
    history = actor.fit(inputs, actions, sample_weight=weights,
                        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)

    # Guardar
    os.makedirs(save_dir, exist_ok=True)
    actor.save(os.path.join(save_dir, "actor_awr.keras"))

    # Gráfica de pérdida
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label="Loss")
    plt.title(f"AWR Training - Beta {beta}, Clip {clip_weight}")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_loss.png"))
    plt.close()
    print(f"✅ Guardado en {save_dir}")

# ==== EJECUTAR TODAS LAS COMBINACIONES ====
for beta in ADV_BETAS:
    for clip in ADV_CLIPS:
        folder_name = f"AWR_beta{beta}_clip{clip}"
        train_awr(beta, clip, os.path.join(OUTPUT_DIR, folder_name))
        # print(folder_name)

print("\n🎉 Todos los experimentos AWR finalizados.")
