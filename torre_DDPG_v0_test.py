
# Script que:
# 1. Carga el modelo del entorno (model_env) desde un archivo .keras.
# 2. Carga el dataset original.
# 3. Ejecuta 100 episodios de 128 pasos cada uno:
    # - Cada episodio inicia desde un índice aleatorio donde reset == True.
    # - Se extraen 120 filas reales consecutivas para comparar contra 120 pasos simulados del modelo.
# 3. Compara estados, acciones y recompensas actuales vs. generadas por política.
# 4. Guarda los resultados simulados y reales en CSV para trazabilidad.
# 5. Crear las siguientes gráficas:
#   - Histogramas por variable (estados y acciones) — actual vs con política.
#   - Tendencias por variable (estados y acciones) — con marcas por episodio.
#   - Histograma de recompensas promedio por hora — actual vs con política

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import csv
from IPython.display import clear_output



# ==== CONFIGURACIÓN ====
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
MODEL_ENV_PATH = 'models/modelo_torre.keras'        # modelo del entorno
ACTOR_PATH =  "checkpoints/DDPGv0/torre_DDPG_actor_1500.keras"  #  'models/DDPGv0/torre_DDPG_actor.keras'  #      política entrenada
REPORT_PATH = 'reports/eval_torre_DDPG_v0.csv'
IMG_DIR = 'img/DDPGv0'

EPISODES = 100
EPISODE_LENGTH = 60 # 120

# Crear carpetas si no existen
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==== CARGA DE DATOS Y MODELOS ====
df = pd.read_csv(DATASET_PATH)
# model_env = tf.keras.models.load_model(MODEL_ENV_PATH)

actor = tf.keras.models.load_model(ACTOR_PATH)


# Cargar modelo del entorno ======================
# model_env = tf.keras.models.load_model(MODEL_ENV_PATH)

# Cargar los cinco modelos del ensamble
ensemble_paths = [
    "models/modelo_torre_f1.keras",
    "models/modelo_torre_f2.keras",
    "models/modelo_torre_f3.keras",
    "models/modelo_torre_f4.keras",
    "models/modelo_torre_f5.keras"
]
model_ensemble = [tf.keras.models.load_model(path) for path in ensemble_paths]

# Función de predicción del ensamble decorada
@tf.function
def ensemble_predict(input_tensor):
    preds = [m(input_tensor, training=False) for m in model_ensemble]
    return tf.reduce_mean(tf.stack(preds), axis=0)

model_env = ensemble_predict

print("Modelos y datos cargados")


# ==== SELECCIÓN DE COLUMNAS ====
action_cols = [f'a{i+1}' for i in range(8)]
external_cols = [f'e{i+1}' for i in range(10)]
state_cols = [f's{i+1}' for i in range(16)]

acciones_real = df[action_cols].values.astype(np.float32)
externas = df[external_cols].values.astype(np.float32)
estados = df[state_cols].values.astype(np.float32)
resets = df['reset'].values.astype(bool)

# ==== BÚSQUEDA DE PUNTOS DE INICIO VÁLIDOS ====
valid_starts = [
    i for i in range(len(df) - EPISODE_LENGTH)
    if np.all(resets[i:i + EPISODE_LENGTH])
]
random.seed(1012)
valid_starts = random.sample(valid_starts, min(EPISODES, len(valid_starts)))
print(f"Se usarán {len(valid_starts)} episodios para evaluación")

# ==== ESTRUCTURAS PARA ALMACENAR RESULTADOS ====
all_real_states = []
all_gen_states = []
all_real_actions = []
all_gen_actions = []
rewards_real = []
rewards_gen = []


# ==== FUNCIÓN DE RECOMPENSA ====
def compute_reward(state, action):
    reward = (
        -0.05 * action[1] +         # Bombeo_Low_Pump_P_401
        -0.05 * state[1] +          # P404_High_Pump_Pressure_CV
        -0.05 * state[5] +          # Tower_Input_Air_Fan_Speed_Feedback
        -0.35 * state[7] +          # Torre_Horno_Flujo_Gas
        -0.05 * state[14] +         # F501_Ciclone_01_Speed
        -0.0125 * state[3] +        # Bombeo_Aero_Boost_FT_0371_Kg_h
        -0.0125 * state[4] +        # Bombeo_Aero_Boost_PT_0371_BAR
        1.2 * state[12]             # Torre_PB_Flujo_Schenck
    )
    return reward

# ==== SIMULACIÓN DE EPISODIOS ====
for ep, start in enumerate(valid_starts):
    # clear_output(wait = True)
    # os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Episodio {ep + 1}/{len(valid_starts)}")

    s = estados[start].copy()
    gen_states = []
    gen_actions = []

    for t in range(EPISODE_LENGTH):
        idx = start + t
        externals_t = externas[idx]
        state_input = np.concatenate([externals_t, s]).reshape(1, -1).astype(np.float32)
        action = actor(state_input)[0].numpy()
        sa_input = np.concatenate([action, externals_t, s]).reshape(1, -1).astype(np.float32)
        s_next = model_env(sa_input)[0].numpy()

        gen_states.append(s_next)
        gen_actions.append(action)
        s = s_next

    all_gen_states.append(gen_states)
    all_gen_actions.append(gen_actions)
    all_real_states.append(estados[start + 1:start + 1 + EPISODE_LENGTH])
    all_real_actions.append(acciones_real[start:start + EPISODE_LENGTH])

    r_real = [compute_reward(s, a) for s, a in
              zip(estados[start + 1:start + 1 + EPISODE_LENGTH], acciones_real[start:start + EPISODE_LENGTH])]
    r_gen = [compute_reward(s, a) for s, a in zip(gen_states, gen_actions)]
    rewards_real.extend(r_real)
    rewards_gen.extend(r_gen)

# Aplanar
real_s = np.array(all_real_states).reshape(-1, 16)
gen_s = np.array(all_gen_states).reshape(-1, 16)
real_a = np.array(all_real_actions).reshape(-1, 8)
gen_a = np.array(all_gen_actions).reshape(-1, 8)

# Guardar CSV
header = [f's{i + 1}' for i in range(16)] + [f'a{i + 1}' for i in range(8)] + ['tipo']
with open(REPORT_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in np.hstack([real_s, real_a]):
        writer.writerow(list(row) + ['actual'])
    for row in np.hstack([gen_s, gen_a]):
        writer.writerow(list(row) + ['politica'])


# Gráficas
def plot_histogram(data_real, data_sim, labels, filename, title_prefix):
    fig, axs = plt.subplots(4, len(labels) // 4, figsize=(18, 10))
    axs = axs.flatten()
    for i, label in enumerate(labels):
        axs[i].hist(data_real[:, i], bins=40, alpha=0.5, label='Actual')
        axs[i].hist(data_sim[:, i], bins=40, alpha=0.5, label='Política')
        axs[i].set_title(f'{title_prefix} {label}')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{filename}")
    plt.close()


plot_histogram(real_s, gen_s, [f's{i + 1}' for i in range(16)], 'eval_hist_estados.png', 'Estado')
plot_histogram(real_a, gen_a, [f'a{i + 1}' for i in range(8)], 'eval_hist_acciones.png', 'Acción')


# Tendencias
def plot_tendencias(data_real, data_sim, labels, filename):
    fig, axs = plt.subplots(len(labels) // 4, 4, figsize=(16, 10))
    axs = axs.flatten()
    for i, label in enumerate(labels):
        axs[i].plot(data_real[:, i], label='Actual', alpha=0.7)
        axs[i].plot(data_sim[:, i], label='Política', alpha=0.7)
        axs[i].set_title(label)
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{filename}")
    plt.close()


plot_tendencias(real_s, gen_s, [f's{i + 1}' for i in range(16)], 'eval_tend_estados.png')
plot_tendencias(real_a, gen_a, [f'a{i + 1}' for i in range(8)], 'eval_tend_acciones.png')

# Histograma de recompensa por hora (cada episodio aporta dos)
reward_per_hour_real = [np.mean(rewards_real[i:i + 60]) for i in range(0, len(rewards_real), 60)]
reward_per_hour_gen = [np.mean(rewards_gen[i:i + 60]) for i in range(0, len(rewards_gen), 60)]

plt.figure(figsize=(10, 6))
plt.hist(reward_per_hour_real, bins=20, alpha=0.6, label='Actual')
plt.hist(reward_per_hour_gen, bins=20, alpha=0.6, label='Política')
plt.title("Distribución de recompensa promedio por hora")
plt.xlabel("Recompensa promedio")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.savefig(f"{IMG_DIR}/eval_recompensa_por_hora.png")
plt.close()

print("Finalización exitosa")