# !pip install gym
# !pip install tqdm
# !pip install gymnasium
# !pip install "gymnasium[toy-text]"

# https://gymnasium.farama.org/environments/toy_text/frozen_lake/

from tqdm import tqdm
import gymnasium as gym
import numpy as np


idx_to_action = {
    0:"<", #left
    1:"v", #down
    2:">", #right
    3:"^" #up
}

# create a single game instance
# env = gym.make("FrozenLake-v1")
# env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode= "human")
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="ansi")
# ansi rgb_array_list  rgb_array  human

# desc=["SFFF", "FHFH", "FFFH", "HFFG"]

# gym("FrozenLake-v1", render_mode= "human")

print("The initial state: ", env.reset())
print(" and it looks like: ")
print(env.render())

print("Now let's take an action: ")
new_state, reward, done, info, probab = env.step(1)
print(env.render())
print("Done:", done, "| Estado:", new_state, "| Premio:", reward, "| info:", info)

# --------------

def avanzar():
    """
    Función para avanzar un paso en el juego
    """
    new_state, reward, done, info, probab = env.step(1)
    print(env.render())
    print("Done:", done, "| Estado:", new_state, "| Premio:", reward, "| info:", info)
    if done:
        env.reset()
        if reward > 0:
            print("Ganaste!")
        else:
            print("Perdiste!")

for i in range(1, 10):
    avanzar()


# --------------
# Policy ----

print("# Policy ----")

n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize random_policy:
def init_random_policy():
    random_policy = {}
    for state in range(n_states):
        random_policy[state] = np.random.choice(n_actions)
    return random_policy

# function to evaluate our policy
def evaluate(env, policy, max_episodes = 100):
    tot_reward = 0
    for ep in range(max_episodes):
        state = env.reset()
        state = 0
        done = False
        ep_reward = 0
        # Reward per episode
        while not done:
            action = policy[state]
            new_state, reward, done, info, probab = env.step(action)
            ep_reward += reward
            state = new_state
            if done:
                tot_reward += ep_reward
    return tot_reward/(max_episodes)

# Politica random y su evaluación
politica_random = init_random_policy()
evaluate(env, politica_random)

# Looking for the best policy: Random search
best_policy = None
best_score = -float('inf')

# Random search
for i in range(1, 10): #tip: you can use tqdm(range(1,10000)) for a pro
    policy = init_random_policy()
    score = evaluate(env,policy,100)
    if score > best_score:
        best_score = score
        best_policy = policy
    if i%5000 == 0:
        print("Best score:", best_score)

print("Best score:", best_score, "| Best policy:")
print(best_policy)


# --------------
# Let's see the policy in action
def play(env, policy, render = False):
    s = env.reset()
    s = 0
    d = False
    while not d:
        a = policy[s]
        print("*"*10)
        print("State: ",s)
        print("Action: ",idx_to_action[a])
        s, r, d, i, p  = env.step(a)
        if render:
            env.render()
        if d:
            print(r)

# Let’s create a small function to print a nicer policy:
def print_policy(policy):
    lake = "SFFFFHFHFFFHHFFG"
    arrows = [idx_to_action[policy[i]]
    if lake[i] in 'SF' else '*' for i in range(n_states)]
    for i in range(0,16,4):
        print(''.join(arrows[i:i+4]))

print_policy(best_policy)
play(env, best_policy)


# Using a different policy
# theta = 0.25*np.ones((n_states,n_actions))
def random_parameter_policy(theta):
    theta = theta/np.sum(theta, axis=1, keepdims = True) # ensure that the
    policy = {}
    probs = {}
    for state in range(n_states):
        probs[state] = np.array(theta[state, :])
        policy[state] = np.random.choice(n_actions, p = probs[state])
    return policy

best_policy = None
best_score = -float('inf')
alpha = 1e-2

# Random search
for i in range(1, 10):
    theta = 0.25 * np.ones((n_states, n_actions))
    policy = random_parameter_policy(theta)
    score = evaluate(env, policy, 100)
    if score > best_score:
        best_score = score
        best_policy = policy
    theta = theta + alpha*(score - best_score) * np.ones((n_states, n_actions))
    if i%5000 == 0:
        print("Best score:", best_score)

print("Best score:", best_score, "| Best policy:")
print(best_policy)
print_policy(best_policy)


# --------------
# Pruebas para acomodar los datos

import matplotlib
import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Dense
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# matplotlib.use('Agg')  # Configuración para entornos no interactivos
# import matplotlib.pyplot as plt

# ========= CONFIGURACIÓN =========
EPOCHS = 4
VALIDATION_SPLIT = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 137
LEARNING_RATE = 0.0001


data = pd.read_csv('data/datos_Normal_v2_26abr_V1Filter.csv')

if 'time_stamp' in data.columns:
    data = data.drop(columns=['time_stamp'])

list_cols = data.columns.tolist()

# Columnas que NO son predictores (variables de control/configuración)
list_no_predict = ['Number_of_Jets_Open', 'Bombeo_Low_Pump_P_401', 'P404_High_Pump_Pressure_SP',
                   'Apertura_Valvula_Flujo_Aeroboost_FCV_0371', 'Apertura_Valvula_Presion_Aeroboost',
                   'Tower_Input_Air_Fan_Speed_Ref', 'Tower_Input_Temperature_SP', 'Tower_Internal_Pressure_SP']

# Columnas a predecir
list_PREDICT = [col for col in list_cols if col not in list_no_predict]

# ========= DIVISIÓN DE DATOS =========
print("\nDividiendo datos...")
train_data = data.sample(frac=1-TEST_SIZE, random_state=RANDOM_STATE)
test_data = data.drop(train_data.index)


# Reemplazar por:
train_data = data.iloc[0:-1].sample(frac=1-TEST_SIZE, random_state=RANDOM_STATE)
test_data = data.iloc[0:-1].drop(train_data.index)

train_data_y = data.iloc[train_data.index + 1]
train_data_y = train_data_y[list_PREDICT]

test_data_y = data.iloc[test_data.index + 1]
test_data_y = test_data_y[list_PREDICT]


print(f"Tamaño del conjunto de entrenamiento: {len(train_data)}")
print(f"Tamaño del conjunto de prueba: {len(test_data)}")


train_data[list_cols]
train_data.iloc[train_data.index, list_PREDICT]

[train_data.index + 1] in range(len(data))

len(data) in train_data.index 


# ========= ENTRENAMIENTO =========
print("\nIniciando entrenamiento...")
history = model.fit(train_data[list_cols],
                    train_data[list_PREDICT],
                    epochs=EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1)
