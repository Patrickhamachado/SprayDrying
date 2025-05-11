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
OUTPUT_DIR = "models/AWR_Plus/Grid_10epocas"

NUM_EPOCHS = 2 # 100
BATCH_SIZE = 128  # 16 # 64
SEED = 5292

ADV_BETAS = [0.5, 0.3]           # [0.2, 0.5, 1.0]
ADV_CLIPS = [5.0] #  [1.0, 2.5, 5.0]

# ==== SEMILLA ====
def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# ==== Replay Buffer simple ====
class SimpleReplayBuffer:
    def __init__(self, states, externals, actions):
        self.inputs = np.concatenate([externals, states], axis=1)
        self.actions = actions

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.inputs), size=batch_size, replace=False)
        return self.inputs[idxs], self.actions[idxs], idxs



# ==== ENTRENAMIENTO AWR+ ====
def train_awr_plus(beta, clip, save_dir):
    print(f"\n🔧 Entrenando AWR+ con beta={beta}, clip={clip}")
    set_seed(SEED)

    # ==== Función de predicción del crítico ====
    @tf.function
    def critic_infer(inputs):
        return critic(inputs, training=False)

    # ==== Recompensa estimada ====
    def get_advantages(critic, inputs, actions, beta, clip):
        # print("Estimando recompensas...")
        q_vals = critic_infer([inputs, actions]).numpy().squeeze()

        # baseline = np.mean(q_vals)                                            # Original
        baseline = critic_infer([inputs, actor(inputs)]).numpy().squeeze()      # Lo que haría el actor


        adv = q_vals - baseline
        weights = np.exp(adv / beta)
        if clip is not None:
            weights = np.clip(weights, 0, clip)
        return adv.astype(np.float32), weights.astype(np.float32)

    print("Cargando datos...")
    df = pd.read_csv(CSV_PATH)
    df = df[df['reset'] == 1].reset_index(drop=True)

    state_cols = [f"s{i+1}" for i in range(16)]
    external_cols = [f"e{i+1}" for i in range(10)]
    action_cols = [f"a{i+1}" for i in range(8)]

    states = df[state_cols].values.astype(np.float32)
    externals = df[external_cols].values.astype(np.float32)
    actions = df[action_cols].values.astype(np.float32)

    inputs = np.concatenate([externals, states], axis=1)

    print("Cargando modelos Crítico y Actor...")
    actor = tf.keras.models.load_model(WARMUP_ACTOR_PATH)
    critic = tf.keras.models.load_model(WARMUP_CRITIC_PATH)

    print("Creando buffer...")
    buffer = SimpleReplayBuffer(states, externals, actions)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = tf.keras.losses.MeanSquaredError()
    losses = []

    print("Iniciando épocas...")
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        for _ in range(len(inputs) // BATCH_SIZE):
            x, y, idxs = buffer.sample(BATCH_SIZE)

            # Cargando recompensas
            adv, weights = get_advantages(critic, x, y, beta, clip)

            with tf.GradientTape() as tape:

                # Actualizando actor
                pred = actor(x, training=True)
                loss = tf.reduce_mean(weights * tf.reduce_sum(tf.square(y - pred), axis=1))
            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            epoch_losses.append(loss.numpy())
        losses.append(np.mean(epoch_losses))
        print(f"Época {epoch+1}/{NUM_EPOCHS} - Loss: {losses[-1]:.6f}")

    # Guardar resultados
    os.makedirs(save_dir, exist_ok=True)
    actor.save(os.path.join(save_dir, "actor_awr.keras"))

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss")
    plt.title(f"AWR+ Training - Beta {beta}, Clip {clip}")
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
        folder_name = f"AWR_plus_beta{beta}_clip{clip}"
        path = os.path.join(OUTPUT_DIR, folder_name)
        train_awr_plus(beta, clip, path)

print("\n🎉 Todos los experimentos AWR+ finalizados.")
