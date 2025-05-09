
# Construcción de un modelo DDPG, Deep Deterministic Policy Gradient, para optimizar la operación de la
# torre de secado

# Estructura general del script
# 1. Cargar CSV y construir el conjunto de datos.
# 2. Calcular la recompensa con una función reward_fn(s, a).
# 3. Definir OfflineEnv, un entorno Gym simulado a partir del dataset.
# 4. Crear el entorno
# 5. Construir el agente DDPG: Actor y Critic
# 6. Replay Buffer
# 7. Llenar el buffer desde el dataset
# 8. Función de entrenamiento del agente
# 9. Entrenamiento
# 10. Guardar modelo


# Librerías ===========================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
import gym
from gym import spaces
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import os
import csv

# Parte 1: Carga del CSV y preparación del dataset  =====================================

# Rutas y configuraciones
CSV_PATH =  "data/datos_Normal_a_e_s_7may.csv"
PLOT_PATH = "img"              # Guardar gráficas en esta ruta
REPORT_PATH = "reports"                   # # Guardar reportes en esta ruta
NUM_EPISODES = 5000               # Número de episodios a entrenar 5000
CHECKPOINT_INTERVAL = 500           # Cada cuanto guardar los modelos  500
PLOT_INTERVAL = 500              # Cada cuántos episodios guardar gráficas 500
BATCH_EPISODE = 8             # Número de batches por episodio 128
BUFFER_SAMPLE = 64              # Támaño de la muestra de cada batch
MODEL_PATH = "models"
MODEL_ENV_PATH = 'models/modelo_torre.keras'

BUFFER_SIZE = 200000            # Tamaño del buffer circular con el que se entrenan los modelos
NUM_REAL_INIT = 100000          # Tamaño de inicialización del buffer con datos reales
MAX_ENV_STEPS = 128             # Tamaño máximo de métodos env.step(accion) consecutivos
CRITIC_LR = 1e-4                # Learning rate del modelo Crítico
ACTOR_LR = 1e-4                 # Learning rate del modelo Actor

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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


# Cargar modelos Actor - Crítico ======================

# actor, critic, target_actor, target_critic, counter_val = load_checkpoint()
# train_step.counter.assign(counter_val)



# 8. Función de entrenamiento del agente =======================================================
actor_update_freq = 2

@tf.function
def train_step(states, externals, actions, rewards, next_states, next_externals, dones):
    # Concatenar externals y states
    states_input = tf.concat([externals, states], axis=1)
    next_states_input = tf.concat([next_externals, next_states], axis=1)
    # critic.trainable = False

    # Entrenamiento del crítico
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_states_input)

        # Valor objetivo: y = r + γ * Q'(s', a')
        target_q = tf.squeeze(target_critic([next_states_input, target_actions]), axis=1)
        y = rewards + gamma * (1.0 - tf.cast(dones, tf.float32)) * target_q

        # Valor estimado por el crítico actual
        critic_value = tf.squeeze(critic([states_input, actions]), axis = 1)
        # Pérdida del crítico
        critic_loss = tf.keras.losses.MSE(y, critic_value)

    # Gradiente y actualización del crítico
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # Entrenamiento del actor con menor frecuencia
    actor_loss = tf.constant(0.0)  # inicialización segura
    if tf.equal(train_step.counter % actor_update_freq, 0):
        with tf.GradientTape() as tape:
            actions_pred = actor(states_input)
            q_val = critic([states_input, actions_pred])
            actor_loss = -tf.reduce_mean(q_val)
        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    # critic.trainable = True

    # Actualización suave de las redes objetivo
    for target_param, param in zip(target_critic.trainable_variables, critic.trainable_variables):
        target_param.assign(tau * param + (1 - tau) * target_param)
    for target_param, param in zip(target_actor.trainable_variables, actor.trainable_variables):
        target_param.assign(tau * param + (1 - tau) * target_param)

    # Actualización del contador
    train_step.counter.assign_add(1)

    return critic_loss, actor_loss

# Inicializar contador como variable de TensorFlow
train_step.counter = tf.Variable(0)


# Verifica si existe checkpoint para continuar
checkpoint_files = [
    "torre_DDPG_actor.keras",
    "torre_DDPG_critic.keras",
    "torre_DDPG_target_actor.keras",
    "torre_DDPG_target_critic.keras",
    "train_step_counter.txt"
]

checkpoint_exists = all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in checkpoint_files)


# =================== CARGAR MODELOS Y CONTADOR ============================
def load_checkpoint(path="models"):
    actor = tf.keras.models.load_model(f"{path}/torre_DDPG_actor.keras")
    critic = tf.keras.models.load_model(f"{path}/torre_DDPG_critic.keras")
    target_actor = tf.keras.models.load_model(f"{path}/torre_DDPG_target_actor.keras")
    target_critic = tf.keras.models.load_model(f"{path}/torre_DDPG_target_critic.keras")
    with open(f"{path}/train_step_counter.txt", "r") as f:
        counter_val = int(f.read().strip())
    train_step.counter.assign(counter_val)
    return actor, critic, target_actor, target_critic, counter_val



if checkpoint_exists:
    print("🟢 Checkpoint encontrado. Cargando modelos y contador...")
    actor, critic, target_actor, target_critic, counter_val = load_checkpoint()

    train_step.counter.assign(counter_val)
else:
    print("🔵 No se encontró checkpoint. Entrenando desde cero.")


# Crear carpeta de resultados si no existe
os.makedirs(PLOT_PATH, exist_ok = True)

# Cargar CSV con pandas
df = pd.read_csv(CSV_PATH)
print(f"Datos leídos: {df.shape[0]} filas x {df.shape[1]} columnas")
print(df.head(1))


# Separar nombres de columnas
action_cols = [f'a{i+1}' for i in range(8)]           # a1–a8
external_cols = [f'e{i+1}' for i in range(10)]         # e1–e10
state_cols = [f's{i+1}' for i in range(16)]            # s1–s16

# Convertir a numpy arrays
actions = df[action_cols].values.astype(np.float32)
externals = df[external_cols].values.astype(np.float32)
states = df[state_cols].values.astype(np.float32)
dones = df['done'].values.astype(bool)
resets = df['reset'].values.astype(bool)
valid_eval_indices = np.where(resets[:-1])[0]

print("Shapes esperadas:")
print("Actions: (x, 8): ", actions.shape)
print("Externals: (x, 10): ", externals.shape)
print("States: (x, 16): ", states.shape)


# Historiales para las gráficas de entrenamiento
actor_losses = []
critic_losses = []
reward_true_history = []
reward_pred_history = []
env_model_mse_history = []
env_model_mse_per_var = [[] for _ in range(16)]
flujo_schenck_true = []
flujo_schenck_pred = []
humedad_true = []
humedad_pred = []


# 1.2. Funciones varias =================================================================

# Moving average para gráficas de pérdidas
def moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode = 'valid')


# =================== GUARDAR MODELOS Y CONTADOR ============================
def save_checkpoint(actor, critic, target_actor, target_critic, counter, path="models", epoca = ""):
    actor.save(f"{path}/torre_DDPG_actor{epoca}.keras")
    critic.save(f"{path}/torre_DDPG_critic{epoca}.keras")
    target_actor.save(f"{path}/torre_DDPG_target_actor{epoca}.keras")
    target_critic.save(f"{path}/torre_DDPG_target_critic{epoca}.keras")
    with open(f"{path}/train_step_counter{epoca}.txt", "w") as f:
        f.write(str(counter.numpy()))


# 2. Función de recompensa dinámica ======================================================
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

# Prueba de la función de recompensa
# sample_idx = 10
# sample_state = states[sample_idx]
# sample_action = actions[sample_idx]
# reward_test = compute_reward(sample_state, sample_action)
# print(f"🔍 Recompensa de prueba en fila {sample_idx}: {reward_test:.4f}")
# print("Estados:")
# print(sample_state)
# print("Acciones:")
# print(sample_action)

# 3. Definir HybridEnv, un entorno Gym simulado a partir del dataset =======================
class HybridEnv(gym.Env):
    def __init__(self, states, externals, model_env, reward_fn, reset_mask, dones, max_steps = MAX_ENV_STEPS):
        super().__init__()
        self.states = states
        self.externals = externals
        self.model_env = model_env
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.reset_mask = reset_mask
        self.dones = dones
        self.valid_reset_indices = np.where(reset_mask)[0]

        self.idx = 0
        self.step_count = 0
        self.state = None

    def reset(self):
        self.idx = np.random.choice(self.valid_reset_indices)
        self.state = self.states[self.idx].copy()
        assert self.state.shape[0] == 16, f"Estado inválido con forma {self.state.shape}"
        self.step_count = 0
        return self.state

    def step(self, action):
        external_vars = self.externals[self.idx]
        assert external_vars.shape[0] == 10, f"Externals inválido con forma {external_vars.shape}"

        # Crear entrada para el modelo del entorno
        # input_sa = np.concatenate([action, external_vars, self.state], axis=0).reshape(1, -1)
        input_sa = np.concatenate([
            action.reshape(-1),  # 8
            external_vars,  # 10
            self.state  # 16
        ], axis=0).reshape(1, -1)

        next_state = self.model_env(input_sa)[0].numpy()
        # Calcular recompensa
        reward = self.reward_fn(next_state, action)

        # Actualizar
        self.idx += 1
        self.state = next_state
        self.step_count += 1
        # Terminar si se alcanza el máximo de pasos o si la planta se detuvo
        done = (self.step_count >= self.max_steps) or self.dones[self.idx]

        return next_state, reward, done


# 4. Crear el entorno ===========================================================================
env = HybridEnv(states, externals, model_env, compute_reward, reset_mask = resets, dones = dones)


# 5. Construir el agente DDPG: Actor y Critic =====================================================
def get_actor():
    inputs = layers.Input(shape=(26,))
    out = layers.Dense(256, activation='relu')(inputs)
    out = layers.Dense(256, activation='relu')(out)
    outputs = layers.Dense(8, activation='sigmoid')(out)  # Acciones en [0,1]
    return tf.keras.Model(inputs, outputs)

def get_critic():
    state_input = layers.Input(shape=(26,))
    action_input = layers.Input(shape=(8,))
    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(256, activation='relu')(concat)
    out = layers.Dense(256, activation='relu')(out)
    outputs = layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs)

# actor = get_actor()
# critic = get_critic()
# target_actor = tf.keras.models.clone_model(actor)
# target_critic = tf.keras.models.clone_model(critic)

if checkpoint_exists:
    print("🟢 Checkpoint encontrado. Cargando modelos y contador...")
    actor, critic, target_actor, target_critic, counter_val = load_checkpoint()
    train_step.counter.assign(counter_val)
else:
    print("🔵 No se encontró checkpoint. Cargando modelos warmup...")
    actor = tf.keras.models.load_model("models/warmup_actor.keras")
    critic = tf.keras.models.load_model("models/warmup_critic.keras")
    target_actor = tf.keras.models.clone_model(actor)
    target_critic = tf.keras.models.clone_model(critic)


# 6. Replay Buffer ==========================================================================
class ReplayBuffer:
    def __init__(self, max_size = BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.is_synthetic = []
        self.externals = []
        self.next_externals = []

    def add(self, state, external, action, reward, next_state, next_external, done, synthetic=False):
        if self.size < self.max_size:
            self.states.append(state)
            self.externals.append(external)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.next_externals.append(next_external)
            self.dones.append(done)
            self.is_synthetic.append(synthetic)
            self.size += 1
        else:
            self.states[self.ptr] = state
            self.externals[self.ptr] = external
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.next_externals[self.ptr] = next_external
            self.dones[self.ptr] = done
            self.is_synthetic[self.ptr] = synthetic
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        idxs = random.sample(range(self.size), batch_size)
        return (
            np.array([self.states[i] for i in idxs], dtype=np.float32),
            np.array([self.externals[i] for i in idxs], dtype=np.float32),
            np.array([self.actions[i] for i in idxs], dtype=np.float32),
            np.array([self.rewards[i] for i in idxs], dtype=np.float32),
            np.array([self.next_states[i] for i in idxs], dtype=np.float32),
            np.array([self.next_externals[i] for i in idxs], dtype=np.float32),
            np.array([self.dones[i] for i in idxs], dtype=bool),
            np.array([self.is_synthetic[i] for i in idxs], dtype=bool),
        )

    def get_balance(self):
        synthetic_count = sum(self.is_synthetic)
        real_count = self.size - synthetic_count
        return real_count, synthetic_count

buffer = ReplayBuffer(BUFFER_SIZE)
# buffer = ReplayBuffer(size=len(states) - 1)
gamma = 0.99
tau = 0.005
actor_optimizer = tf.keras.optimizers.Adam(learning_rate = ACTOR_LR)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate = CRITIC_LR)

# 7. Llenar el buffer desde el dataset ========================================================

# Seleccionar índices aleatorios para el inicio del buffer
init_indices = np.random.choice(len(states) - 1, size = NUM_REAL_INIT, replace = False)
for i in init_indices:
    s = states[i]
    e = externals[i]
    a = actions[i]
    s2 = states[i + 1]
    e2 = externals[i + 1]
    r = compute_reward(s2, a)
    d = dones[i]
    buffer.add(s, e, a, r, s2, e2, d, synthetic=False)


# Actualización suave (Polyak averaging) para los modelos =======================================================
def update_target_weights(model, target_model, tau = 0.005):
    for target_param, param in zip(target_model.trainable_variables, model.trainable_variables):
        target_param.assign(tau * param + (1.0 - tau) * target_param)



# 9. Entrenamiento ===============================================================================
for episode in range(NUM_EPISODES):
    state = env.reset()

    episode_reward = 0
    for batch_episode in range(BATCH_EPISODE):  # 100 batches por episodio
        # índice actual del entorno
        idx = env.idx

        external_vars = externals[env.idx]
        state_input = np.concatenate([external_vars, env.state], axis=0).reshape(1, -1).astype(np.float32)

        # Se predice la acción con la red actor
        state_input_tensor = tf.convert_to_tensor(state_input, dtype=tf.float32)
        action = actor(state_input_tensor)[0].numpy().astype(np.float32)

        # Interacción con el entorno
        next_state, reward, done = env.step(action)

        # Almacenar experiencia en el buffer
        buffer.add(state, externals[idx], action, reward, next_state, externals[idx + 1], done, synthetic=True)

        # Avanzar el estado
        state = next_state.copy()
        episode_reward += reward

        # Entrenamiento por batch
        if buffer.size > BUFFER_SAMPLE:
            states_b, externals_b, actions_b, rewards_b, next_states_b, next_externals_b, dones_b, _ = buffer.sample(BUFFER_SAMPLE)
            # Convertir a tensores los cinco primeros (excepto is_synthetic)
            batch = [states_b, externals_b, actions_b, rewards_b, next_states_b, next_externals_b, dones_b]
            batch = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch]

            # Se entrenan las redes
            critic_loss, actor_loss = train_step(*batch)

            # Se historizan las pérdidas
            critic_losses.append(critic_loss.numpy())
            actor_losses.append(actor_loss.numpy())

        if done:
            break

    # Evaluación con un solo punto para monitoreo
    idx = np.random.choice(valid_eval_indices)
    state_idx = states[idx]
    true_action = actions[idx]
    external_vars = externals[idx]  # asegúrate de haber cargado esta variable

    state_input = np.concatenate([external_vars, state_idx], axis=0).reshape(1, -1)
    pred_action = actor(state_input)[0].numpy().astype(np.float32)


    input_sa = np.concatenate([pred_action, external_vars, state_idx], axis=0).reshape(1, -1)
    pred_state = model_env(tf.convert_to_tensor(input_sa, dtype=tf.float32))[0].numpy()

    # Calcular reward real y predicho
    reward_true = compute_reward(states[idx + 1], true_action)
    reward_pred = compute_reward(pred_state, pred_action)
    reward_error = np.abs(reward_true - reward_pred).mean()

    # Comparar estado real vs predicho
    true_next_state = states[idx + 1]  # estado real siguiente
    mse = np.mean((pred_state - true_next_state) ** 2)
    env_model_mse_history.append(mse)
    mse_per_variable = np.square(pred_state - true_next_state[:16])

    # Calcular y guardar el MSE por variable
    for i in range(16):
        var_mse = (pred_state[i] - true_next_state[i]) ** 2
        env_model_mse_per_var[i].append(var_mse)

    # Guardar los MSE por variable en un CSV acumulado por episodio
    csv_accum_path = os.path.join(PLOT_PATH, "mse_acumulado.csv")

    # Si es el primer episodio, escribe encabezado
    if episode == 0 and not os.path.exists(csv_accum_path):
        with open(csv_accum_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ["episodio"] + [f"s{i}" for i in range(len(mse_per_variable))]
            writer.writerow(header)
    # Agrega la fila de este episodio
    with open(csv_accum_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [episode + 1] + list(mse_per_variable)
        writer.writerow(row)


    # Guardar métricas
    reward_true_history.append(reward_true)
    reward_pred_history.append(reward_pred)
    flujo_schenck_true.append(true_next_state[12])
    flujo_schenck_pred.append(pred_state[12])
    humedad_true.append(true_next_state[13])
    humedad_pred.append(pred_state[13])

    # actor_losses.append(actor_loss)
    # critic_losses.append(critic_loss)

    # Graficar cada N episodios
    if (episode + 1) % PLOT_INTERVAL == 0:
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))

        axs[0, 0].plot(actor_losses, label = 'Actor Loss', alpha = 0.3)
        axs[0, 0].plot(critic_losses, label = 'Critic Loss', alpha = 0.3)
        axs[0, 0].plot(moving_average(actor_losses), label = 'Actor Loss (MA)', linewidth = 2)
        axs[0, 0].plot(moving_average(critic_losses), label = 'Critic Loss (MA)', linewidth = 2)
        axs[0, 0].set_title("Pérdidas del Actor y Critic")
        axs[0, 0].legend()

        axs[0, 1].plot(reward_true_history, label='Recompensa Real')
        axs[0, 1].plot(reward_pred_history, label='Recompensa Predicha')
        axs[0, 1].set_title("Recompensa real vs predicha")
        axs[0, 1].legend()

        axs[1, 0].plot(flujo_schenck_true, label='Flujo Schenck Real')
        axs[1, 0].plot(flujo_schenck_pred, label='Flujo Schenck Modelo')
        axs[1, 0].set_title("flujo Schenck (estado)")
        axs[1, 0].legend()

        axs[1, 1].plot(humedad_true, label='Humedad Real')
        axs[1, 1].plot(humedad_pred, label='Humedad Modelo')
        axs[1, 1].set_title("Humedad (estado)")
        axs[1, 1].legend()

        axs[2, 0].plot(env_model_mse_history, label='Error MSE')
        axs[2, 0].set_title("Error del modelo de entorno (MSE)")
        axs[2, 0].legend()

        mean_mse_per_var = [np.mean(m) if len(m) > 0 else 0 for m in env_model_mse_per_var]
        axs[2, 1].bar(range(16), mean_mse_per_var)
        axs[2, 1].set_title("MSE por variable del estado (s1 a s16)")
        axs[2, 1].set_xlabel("Índice de variable")
        axs[2, 1].set_ylabel("MSE promedio")

        fig.suptitle(f"Resumen del entrenamiento - Episodio {episode + 1}")
        plt.tight_layout(rect = [0, 0.03, 1, 0.95])
        plt.savefig(f"{PLOT_PATH}/entr_torre_DDPG_{episode + 1:04d}.png")
        plt.close()

    # checkpoint
    if (episode + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(actor, critic, target_actor, target_critic, train_step.counter, path= CHECKPOINT_DIR, epoca = f"_{episode + 1:04d}")

    # Relación de datos reales vs sintéticos
    real_count, synthetic_count = buffer.get_balance()
    print(f"Episodio {episode + 1}, Recompensa total: {episode_reward:.2f}, Minutos al buffer: {batch_episode}, Buffer (real, sintéticos): {real_count, synthetic_count}, {100 * real_count / (real_count + synthetic_count):.1f}% reales")
    if (episode + 1) % 100 == 0:
        print(f"✅ Episodio {episode + 1} completado. Reward acumulado: {episode_reward:.2f}")

#  las 5 variables con mayor error de predicción del modelo de entorno
avg_mse_per_var = [np.mean(m) for m in env_model_mse_per_var]
top_vars = np.argsort(avg_mse_per_var)[-5:][::-1]
print("\nTop 5 variables con mayor error de predicción:")

# Resumen del MSE por variable
for i in top_vars:
    print(f"s{i}: MSE = {avg_mse_per_var[i]:.6f}")


# 10. Resumen final de desempeño ===========================================================

summary = {
    "episodio": list(range(1, NUM_EPISODES + 1)),
    "reward_true": reward_true_history,
    "reward_pred": reward_pred_history,
    "flujo_schenck_real": flujo_schenck_true,
    "flujo_schenck_predicho": flujo_schenck_pred,
    "humedad_real": humedad_true,
    "humedad_predicho": humedad_pred,
}

# Convertir a DataFrame
df_summary = pd.DataFrame(summary)

# Calcular errores (pueden agregarse más si querés)
df_summary["error_flujo_schenck"] = (df_summary["flujo_schenck_real"] - df_summary["flujo_schenck_predicho"]).abs()
df_summary["error_humedad"] = (df_summary["humedad_real"] - df_summary["humedad_predicho"]).abs()
df_summary["error_reward"] = (df_summary["reward_true"] - df_summary["reward_pred"]).abs()

# Guardar en CSV
df_summary.to_csv(f"{REPORT_PATH}/entr_torre_DDPG.csv", index = False)

# Mostrar resumen general en consola
print("\nResumen final del entrenamiento:")
print(df_summary.describe().round(3))



# 10. Guardar modelo final ===============================================================================
# actor.save(f"{MODEL_PATH}_actor.keras")
# critic.save(f"{MODEL_PATH}_critic.keras")
# save_checkpoint(actor, critic, target_actor, target_critic, train_step.counter)
save_checkpoint(actor, critic, target_actor, target_critic, train_step.counter, path= MODEL_PATH, epoca = "")
