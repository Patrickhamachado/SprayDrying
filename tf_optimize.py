import random
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from cal_reward import RewardCalculator

# ========= CONFIGURACIÓN DE HIPERPARÁMETROS =========
HYPERPARAMS = {'gamma': 0.95,
               'epsilon_init': 1.0,
               'epsilon_min': 0.01,
               'epsilon_decay': 0.3,
               'learning_rate': 0.001,
               'batch_size': 32,
               'buffer_size': 2000,
               'layer_sizes': [64, 128, 64],
               'update_target_every': 100}

def set_hyperparams(new_params):
    global HYPERPARAMS
    HYPERPARAMS.update(new_params)

def get_hyperparams():
    return HYPERPARAMS.copy()

class DQNAgent:
    def __init__(self, state_size, action_size):
        params = get_hyperparams()

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=params['buffer_size'])
        self.gamma = params['gamma']
        self.epsilon = params['epsilon_init']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.update_target_every = params['update_target_every']
        self.steps_since_update = 0

        self.reward_calculator = RewardCalculator("pesos.csv")
        self.model = self._build_model(params['layer_sizes'])
        self.target_model = self._build_model(params['layer_sizes'])
        self.update_target_model()

        self.episode_rewards = []
        self.episode_losses = []
        self.epsilon_history = []

    def _build_model(self, layer_sizes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))

        for size in layer_sizes:
            model.add(tf.keras.layers.Dense(size, activation='relu'))

        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mae'])
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        losses = []

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)

            history = self.model.fit(state, target, epochs=1, verbose=0)
            losses.append(history.history['loss'][0])

            self.steps_since_update += 1
            if self.steps_since_update >= self.update_target_every:
                self.update_target_model()
                self.steps_since_update = 0

        avg_loss = np.mean(losses) if losses else 0
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return avg_loss

    def train(self, data, episodes):
        for e in range(episodes):
            state_df = data.sample(1)
            state = state_df.values.reshape(1, self.state_size)

            action = self.act(state)
            next_state_df = data.sample(1)
            next_state = next_state_df.values.reshape(1, self.state_size)

            reward = self.reward_calculator.calculate_reward(state_df.iloc[0])
            done = False

            self.remember(state, action, reward, next_state, done)
            loss = self.replay()

            self.episode_rewards.append(reward)
            self.episode_losses.append(loss)
            self.epsilon_history.append(self.epsilon)

            if e % 50 == 0:
                self.log_progress(e, episodes)

    def log_progress(self, episode, total_episodes):
        avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
        avg_loss = np.mean(self.episode_losses[-50:]) if self.episode_losses else 0

        print(f"\nEpisodio {episode}/{total_episodes}")
        print(f"Recompensa promedio (últimos 50): {avg_reward:.2f}")
        print(f"Pérdida promedio (últimos 50): {avg_loss:.4f}")
        print(f"Epsilon: {self.epsilon:.3f}")

        self.plot_training(episode)

    def plot_training(self, episode):
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(self.episode_rewards)
        plt.title(f'Recompensas - Episodio {episode}')

        plt.subplot(3, 1, 2)
        plt.plot(self.episode_losses)
        plt.title('Pérdida durante el entrenamiento')

        plt.subplot(3, 1, 3)
        plt.plot(self.epsilon_history)
        plt.title('Exploración (Epsilon)')

        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

    def save_model(self, filename):
        self.model.save(filename)

def main(episodes=500, custom_params=None, input_data='sample_data.csv'):
    if custom_params:
        set_hyperparams(custom_params)

    data = pd.read_csv(input_data)
    if 'time_stamp' in data.columns:
        data = data.drop(columns=['time_stamp'])

    calculator = RewardCalculator("pesos.csv")
    try:
        calculator.validate_data(data)
    except ValueError as e:
        print(f"Error en los datos: {str(e)}")
        return None

    agent = DQNAgent(data.shape[1], action_size=5)
    agent.train(data, episodes)

    agent.save_model('dqn_model.h5')
    print("\nEntrenamiento completado. Modelo guardado como 'dqn_model.h5'")

    return agent

if __name__ == "__main__":
    main(episodes=200,
         input_data='datos_Normal_v2_26abr_V1Filter.csv')
