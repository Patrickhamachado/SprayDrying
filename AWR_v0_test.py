
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
from scipy.stats import ttest_ind

# ==== CONFIGURACIÓN ====
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
ACTOR_PATH = 'models/AWR/actor_awr.keras'
CRITIC_PATH = 'models/warmup_critic.keras'
ENSEMBLE_PATHS = [
    'models/modelo_torre_f1.keras',
    'models/modelo_torre_f2.keras',
    'models/modelo_torre_f3.keras',
    'models/modelo_torre_f4.keras',
    'models/modelo_torre_f5.keras'
]
EPISODES = 100
EPISODE_LENGTH = 60
IMG_DIR = 'img/AWR'
REPORT_PATH = 'reports/eval_awr.csv'

# Crear carpetas si no existen
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ==== CARGA DE DATOS Y MODELOS ====
df = pd.read_csv(DATASET_PATH)
actor = tf.keras.models.load_model(ACTOR_PATH)
critic = tf.keras.models.load_model(CRITIC_PATH)
ensemble_models = [tf.keras.models.load_model(p) for p in ENSEMBLE_PATHS]

@tf.function
def ensemble_predict(x):
    preds = [m(x, training=False) for m in ensemble_models]
    return tf.reduce_mean(tf.stack(preds), axis=0)

# ==== COLUMNAS ====
action_cols = [f'a{i+1}' for i in range(8)]
external_cols = [f'e{i+1}' for i in range(10)]
state_cols = [f's{i+1}' for i in range(16)]
actions_real = df[action_cols].values.astype(np.float32)
externals = df[external_cols].values.astype(np.float32)
states = df[state_cols].values.astype(np.float32)
resets = df['reset'].values.astype(bool)

# ==== INICIOS VÁLIDOS ====
valid_starts = [i for i in range(len(df) - EPISODE_LENGTH) if np.all(resets[i:i + EPISODE_LENGTH])]
np.random.seed(1012)
valid_starts = np.random.choice(valid_starts, EPISODES, replace=False)

# ==== FUNCIÓN DE RECOMPENSA ====
def compute_reward(state, action):
    return (
        -0.05 * action[1] + -0.05 * state[1] +
        -0.05 * state[5] + -0.35 * state[7] +
        -0.05 * state[14] + -0.0125 * state[3] +
        -0.0125 * state[4] + 1.2 * state[12]
    )

# ==== EVALUACIÓN ====
all_real_states, all_gen_states = [], []
all_real_actions, all_gen_actions = [], []
rewards_real, rewards_gen = [], []

for ep, start in enumerate(valid_starts):
    s = states[start].copy()
    gen_states, gen_actions = [], []
    for t in range(EPISODE_LENGTH):
        idx = start + t
        ext = externals[idx]
        state_input = np.concatenate([ext, s]).reshape(1, -1).astype(np.float32)
        action = actor(state_input)[0].numpy()
        sa_input = np.concatenate([action, ext, s]).reshape(1, -1).astype(np.float32)
        s_next = ensemble_predict(sa_input)[0].numpy()
        gen_states.append(s_next)
        gen_actions.append(action)
        s = s_next

    real_s = states[start+1:start+1+EPISODE_LENGTH]
    real_a = actions_real[start:start+EPISODE_LENGTH]
    all_gen_states.append(gen_states)
    all_gen_actions.append(gen_actions)
    all_real_states.append(real_s)
    all_real_actions.append(real_a)
    rewards_real.extend([compute_reward(s,a) for s,a in zip(real_s, real_a)])
    rewards_gen.extend([compute_reward(s,a) for s,a in zip(gen_states, gen_actions)])

# ==== GUARDAR RESULTADOS ====
real_s = np.array(all_real_states).reshape(-1, 16)
gen_s = np.array(all_gen_states).reshape(-1, 16)
real_a = np.array(all_real_actions).reshape(-1, 8)
gen_a = np.array(all_gen_actions).reshape(-1, 8)

header = [f's{i+1}' for i in range(16)] + [f'a{i+1}' for i in range(8)] + ['tipo']
with open(REPORT_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in np.hstack([real_s, real_a]):
        writer.writerow(list(row) + ['actual'])
    for row in np.hstack([gen_s, gen_a]):
        writer.writerow(list(row) + ['awr'])

# ==== GRÁFICAS ====
def plot_hist(data_r, data_g, labels, fname, prefix):
    fig, axs = plt.subplots(4, len(labels) // 4, figsize=(18, 10))
    axs = axs.flatten()
    for i, label in enumerate(labels):
        axs[i].hist(data_r[:, i], bins=40, alpha=0.5, label='Real')
        axs[i].hist(data_g[:, i], bins=40, alpha=0.5, label='AWR')
        axs[i].set_title(f'{prefix} {label}')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{fname}")
    plt.close()

# ==== TEST T Y GRÁFICA COMPARATIVA ====
# Calcular recompensa promedio por hora
reward_per_hour_real = [np.mean(rewards_real[i:i + 60]) for i in range(0, len(rewards_real), 60)]
reward_per_hour_gen = [np.mean(rewards_gen[i:i + 60]) for i in range(0, len(rewards_gen), 60)]

# Realizar el t-test de Student
t_stat, p_value = ttest_ind(reward_per_hour_real, reward_per_hour_gen, equal_var=False)

# Mostrar resumen en consola
print(f"\n--- Test de diferencia de medias ---")
print(f"Media actual:   {np.mean(reward_per_hour_real):.3f}")
print(f"Media política: {np.mean(reward_per_hour_gen):.3f}")
print(f"p-valor:        {p_value:.4f}")

# Graficar con anotación del p-valor
plt.figure(figsize=(10, 6))
plt.hist(reward_per_hour_real, bins=20, alpha=0.6, label='Actual')
plt.hist(reward_per_hour_gen, bins=20, alpha=0.6, label='Política')
plt.title("Distribución de recompensa promedio por hora")
plt.xlabel("Recompensa promedio")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.annotate(f'p-valor = {p_value:.4f}',
             xy=(0.95, 0.95), xycoords='axes fraction',
             fontsize=12, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.savefig(f"{IMG_DIR}/eval_recompensa_por_hora.png")
plt.close()


def plot_trends(data_r, data_g, labels, fname):
    fig, axs = plt.subplots(len(labels)//4, 4, figsize=(16,10))
    axs = axs.flatten()
    for i, label in enumerate(labels):
        axs[i].plot(data_r[:, i], label='Real', alpha=0.7)
        axs[i].plot(data_g[:, i], label='AWR', alpha=0.7)
        axs[i].set_title(label)
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/{fname}")
    plt.close()

plot_hist(real_s, gen_s, [f's{i+1}' for i in range(16)], 'eval_hist_estados_awr.png', 'Estado')
plot_hist(real_a, gen_a, [f'a{i+1}' for i in range(8)], 'eval_hist_acciones_awr.png', 'Acción')
plot_trends(real_s, gen_s, [f's{i+1}' for i in range(16)], 'eval_tend_estados_awr.png')
plot_trends(real_a, gen_a, [f'a{i+1}' for i in range(8)], 'eval_tend_acciones_awr.png')

# Recompensas por hora
reward_hour_real = [np.mean(rewards_real[i:i+60]) for i in range(0, len(rewards_real), 60)]
reward_hour_gen = [np.mean(rewards_gen[i:i+60]) for i in range(0, len(rewards_gen), 60)]

plt.figure(figsize=(10, 6))
plt.hist(reward_hour_real, bins=20, alpha=0.6, label='Actual')
plt.hist(reward_hour_gen, bins=20, alpha=0.6, label='Política')
# Líneas verticales para la mediana
median_real = np.median(reward_per_hour_real)
median_gen = np.median(reward_per_hour_gen)
plt.axvline(median_real, color='blue', linestyle='--', linewidth=2, label=f'Mediana actual: {median_real:.2f}')
plt.axvline(median_gen, color='orange', linestyle='--', linewidth=2, label=f'Mediana política: {median_gen:.2f}')

# Título y anotación
plt.title("Distribución recompensa promedio por hora")
plt.xlabel("Recompensa promedio")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.annotate(f'p-valor = {p_value:.4f}',
             xy=(0.95, 0.95), xycoords='axes fraction',
             fontsize=12, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.savefig(f"{IMG_DIR}/eval_recompensa_por_hora_awr.png")
plt.close()
print("✅ Evaluación completada.")
