import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from scipy.stats import ttest_ind

# ==== CONFIGURACIÓN GENERAL ====
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
CRITIC_PATH = 'models/warmup_critic.keras'
ENSEMBLE_PATHS = [
    'models/modelo_torre_f1.keras',
    'models/modelo_torre_f2.keras',
    'models/modelo_torre_f3.keras',
    'models/modelo_torre_f4.keras',
    'models/modelo_torre_f5.keras'
]
# MODELS_DIR = 'models/AWR_Plus/Grid_5_ReLu'          # AWR plus
MODELS_DIR = 'models/AWR/Grid_Normal_E20_c5'          # AWR
EPISODES = 100      # 100
EPISODE_LENGTH = 120 # 120
SUMMARY_PATH = os.path.join(MODELS_DIR, 'resumen_evaluacion.csv')
SEED = 1012


# ==== CARGAR DATOS Y MODELOS ====
df = pd.read_csv(DATASET_PATH)
df = df[df['reset'] == 1].reset_index(drop=True)
actions_real = df[[f'a{i+1}' for i in range(8)]].values.astype(np.float32)
externals = df[[f'e{i+1}' for i in range(10)]].values.astype(np.float32)
states = df[[f's{i+1}' for i in range(16)]].values.astype(np.float32)
resets = df['reset'].values.astype(bool)

valid_starts = [i for i in range(len(df) - EPISODE_LENGTH) if np.all(resets[i:i + EPISODE_LENGTH])]
np.random.seed(SEED)
valid_starts = np.random.choice(valid_starts, EPISODES, replace=False)

ensemble_models = [tf.keras.models.load_model(p) for p in ENSEMBLE_PATHS]
@tf.function
def ensemble_predict(x):
    preds = [m(x, training=False) for m in ensemble_models]
    return tf.reduce_mean(tf.stack(preds), axis=0)

def compute_reward(state, action):
    return (
        -0.05 * action[1] + -0.05 * state[1] +
        -0.05 * state[5] + -0.35 * state[7] +
        -0.05 * state[14] + -0.0125 * state[3] +
        -0.0125 * state[4] + 1.2 * state[12]
    )

# ==== LISTA PARA EL RESUMEN ====
summary_results = []

# ==== EVALUACIÓN INDIVIDUAL ====
def evaluar_modelo(actor_path, output_dir, label):
    print(f"🔍 Evaluando modelo en {label}")
    actor = tf.keras.models.load_model(actor_path)

    all_real_states, all_gen_states = [], []
    all_real_actions, all_gen_actions = [], []
    rewards_real, rewards_gen = [], []

    for start in valid_starts:
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

    real_s = np.array(all_real_states).reshape(-1, 16)
    gen_s = np.array(all_gen_states).reshape(-1, 16)
    real_a = np.array(all_real_actions).reshape(-1, 8)
    gen_a = np.array(all_gen_actions).reshape(-1, 8)

    # ==== CSV ====
    csv_path = os.path.join(output_dir, 'eval_awr.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f's{i+1}' for i in range(16)] + [f'a{i+1}' for i in range(8)] + ['tipo']
        writer.writerow(header)
        for row in np.hstack([real_s, real_a]):
            writer.writerow(list(row) + ['actual'])
        for row in np.hstack([gen_s, gen_a]):
            writer.writerow(list(row) + ['awr'])

    # ==== Recompensa promedio por hora ====
    rph_real = [np.mean(rewards_real[i:i+60]) for i in range(0, len(rewards_real), 60)]
    rph_gen = [np.mean(rewards_gen[i:i+60]) for i in range(0, len(rewards_gen), 60)]
    t_stat, p_value = ttest_ind(rph_real, rph_gen, equal_var=False)
    median_real = np.median(rph_real)
    median_gen = np.median(rph_gen)

    # Guardar valores para resumen
    summary_results.append({
        "modelo": label,
        "reward_media_real": np.mean(rph_real),
        "reward_media_awr": np.mean(rph_gen),
        "reward_mediana_real": median_real,
        "reward_mediana_awr": median_gen,
        "p_value": p_value
    })

    # ==== Gráfica principal ====
    plt.figure(figsize=(10,6))
    plt.hist(rph_real, bins=20, alpha=0.6, label='Real')
    plt.hist(rph_gen, bins=20, alpha=0.6, label='AWR')
    plt.axvline(median_real, color='blue', linestyle='--', linewidth=2, label=f'Mediana real: {median_real:.2f}')
    plt.axvline(median_gen, color='orange', linestyle='--', linewidth=2, label=f'Mediana AWR: {median_gen:.2f}')
    plt.annotate(f'p-valor = {p_value:.4f}', xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=12, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title("Distribución recompensa promedio por hora")
    plt.xlabel("Recompensa promedio")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_recompensa_por_hora.png"))
    plt.close()

    # ==== Otras gráficas ====
    def plot_hist(data_r, data_g, labels, fname, prefix):
        fig, axs = plt.subplots(4, len(labels)//4, figsize=(18, 10))
        axs = axs.flatten()
        for i, label in enumerate(labels):
            axs[i].hist(data_r[:, i], bins=40, alpha=0.5, label='Real')
            axs[i].hist(data_g[:, i], bins=40, alpha=0.5, label='AWR')
            axs[i].set_title(f'{prefix} {label}')
            axs[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    def plot_trend(data_r, data_g, labels, fname):
        fig, axs = plt.subplots(len(labels)//4, 4, figsize=(16,10))
        axs = axs.flatten()
        for i, label in enumerate(labels):
            axs[i].plot(data_r[:, i], label='Real', alpha=0.7)
            axs[i].plot(data_g[:, i], label='AWR', alpha=0.7)
            axs[i].set_title(label)
            axs[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    plot_hist(real_s, gen_s, [f's{i+1}' for i in range(16)], 'eval_hist_estados.png', 'Estado')
    plot_hist(real_a, gen_a, [f'a{i+1}' for i in range(8)], 'eval_hist_acciones.png', 'Acción')
    plot_trend(real_s, gen_s, [f's{i+1}' for i in range(16)], 'eval_tend_estados.png')
    plot_trend(real_a, gen_a, [f'a{i+1}' for i in range(8)], 'eval_tend_acciones.png')

    print(f"✅ Evaluación finalizada para: {label}\n")


# ==== EJECUTAR SOBRE TODAS LAS CARPETAS ====
for folder_name in sorted(os.listdir(MODELS_DIR)):
    folder_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    actor_model_path = os.path.join(folder_path, 'actor_awr.keras')
    if os.path.exists(actor_model_path):
        evaluar_modelo(actor_model_path, folder_path, label=folder_name)
    else:
        print(f"⚠️ No se encontró actor_awr.keras en {folder_path}")

# ==== GUARDAR RESUMEN ====
df_summary = pd.DataFrame(summary_results)
df_summary.to_csv(SUMMARY_PATH, index=False)
print(f"\n📊 Resumen guardado en: {SUMMARY_PATH}")

