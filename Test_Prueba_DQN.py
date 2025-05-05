import numpy as np
import gymnasium as gym
from tensorflow.keras.models import load_model

# Cargar el entorno (modifica según tu entorno)
env = gym.make("CartPole-v1", render_mode = "human")  # reemplaza por tu entorno

# Cargar el modelo entrenado
model = load_model("models/dqn_model.keras")
model.summary()

# Función para elegir acción usando el modelo
def choose_action(state):
    state = np.array(state).reshape(1, -1)  # Asegura forma (1, estados)
    q_values = model.predict(state, verbose = 0)
    return np.argmax(q_values[0])  # Elegir acción con mayor Q

# Ejecutar 10 episodios
maximo = 0
promedio = 0

for i in range(10):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = choose_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Total reward {i + 1}: {total_reward}")
    promedio += total_reward / 10
    maximo = max(maximo, total_reward)

print(f"Máximo: {maximo}, Promedio: {round(promedio, 1)}")