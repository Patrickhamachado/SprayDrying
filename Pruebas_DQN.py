
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import os
import math

MODEL_PATH = "models/dqn_model.keras"

# Basado en
# https://github.com/maciejbalawejder/Reinforcement-Learning-Collection/blob/main/DQN/DQN.ipynb


# Defining models ===========================

class QNetwork:
    def __init__(self, input_dim, output_dim, lr, load_existing = False, model_path = MODEL_PATH):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        if load_existing and os.path.exists(model_path):
            print("📂 Cargando modelo existente para Qpolicy y Qtarget...")
            self.Qpolicy = tf.keras.models.load_model(model_path)
            self.Qtarget = tf.keras.models.load_model(model_path)
        else:
            print("🆕 Creando nuevos modelos para Qpolicy y Qtarget...")
            self.Qpolicy = self.create()
            self.Qtarget = self.create()
            self.Qtarget.set_weights(self.Qpolicy.get_weights())

    def create(self):
        model = Sequential()
        # Corrección
        model.add(tf.keras.Input(shape=(self.input_dim,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation = 'relu'))
        model.add(Dense(128, activation = 'relu'))
        model.add(Dense(self.output_dim, activation = 'linear'))
        model.compile(optimizer = RMSprop(learning_rate = self.lr, rho = 0.95, epsilon = 0.01), loss = "mse", metrics = ['accuracy'])
        return model


class DQNAgent:
    def __init__(self, lr=2.5e-4, gamma=0.99, epsilon=1, decay_coe=0.99975, min_eps=0.001,
                 batch_size=64, memory_size=10_000, episodes=5_000, C=5):

        self.env = gym.make('CartPole-v1')
        # env = gym.make('CartPole-v1', render_mode="human")

        self.states = len(self.env.observation_space.low)
        self.n_actions = self.env.action_space.n

        # self.states = len(env.observation_space.low)
        # self.n_actions = env.action_space.n

        self.actions = [i for i in range(self.n_actions)]

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_coe = decay_coe
        self.min_eps = min_eps
        self.episodes = episodes
        self.batch_size = batch_size
        self.D = deque(maxlen=memory_size)  # replay memory
        self.C = C

        self.terminal_state = False  # end of the episode
        self.target_counter = 0

        # Plot data
        self.timestep = self.episodes / 10
        self.history = []
        self.reward_data = []
        self.epsilon_data = []

        # Si existe, carga el modelo existente para continuar con el entrenamiento
        self.model = QNetwork(self.states, self.n_actions, self.lr,
                              load_existing =os.path.exists(MODEL_PATH),
                              model_path = MODEL_PATH)

        # Smooth epsilon
        self.a = 0.35
        self.b = 0.1
        self.c = 0.01

    def state_shape(self, states):
        # Corrección
        return np.array(states).reshape(1, self.states)
        # Fin Corrección
        # states = np.array(states)
        # if len(states.shape) == 1:
        #     return states.reshape(1, -1)  # estado individual
        # return states.reshape(-1, *states.shape)


    def decrement_epsilon(self, time):
        '''
        if self.epsilon > self.min_eps:
            self.epsilon *= self.decay_coe
        else:
            self.epsilon = self.min_eps
        '''
        s_time = (time - self.a * self.episodes) / (self.b * self.episodes)
        cosh = np.cosh(math.exp(-s_time))
        self.epsilon = 1 - (1 / cosh + (time * self.c / self.episodes))

    def update_D(self, s, a, r, s_, done):
        # self.D.append([self.state_shape(s), a, r, self.state_shape(s_), done])
        self.D.append([np.array(s), a, r, np.array(s_), done])

    def choose_action(self, states):
        if np.random.random() > (1 - self.epsilon):
            action = np.random.choice(self.actions)
        else:
            states = self.state_shape(states)
            action = np.argmax(self.model.Qpolicy.predict(states, verbose = 0))

        return action

    def minibatch(self):
        return random.sample(self.D, self.batch_size)

    def graphs(self, episode):
        f1 = plt.figure(1)
        plt.plot([i for i in range(len(self.reward_data))], self.reward_data)
        plt.ylabel('Score per episode')
        plt.xlabel('Episodes')
        plt.savefig(r'img/dqn_reward_e{}v2.png'.format(episode), dpi=500)

        f2 = plt.figure(2)
        plt.plot([i for i in range(len(self.epsilon_data))], self.epsilon_data)
        plt.ylabel('Epsilon')
        plt.xlabel('Episodes')
        plt.savefig(r'img/dqn_epsilon_e{}v2.png'.format(episode), dpi=500)

        f3 = plt.figure(3)
        plt.plot([i for i in range(len(self.history))], self.history)
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.savefig(r'img/dqn_loss_e{}v2.png'.format(episode), dpi=500)

    def train(self):
        # X - states passed to the NN, y - target

        X, y = [], []

        if len(self.D) >= self.batch_size:
        # Recomendación: empezar e entrenar si el atmaño es grande:
        # if len(self.D) >= self.batch_size and len(self.D) > TRAIN_START::
            SARS = self.minibatch()

            # Corrección
            # s = self.state_shape([row[0] for row in SARS])
            # s = np.squeeze(np.array([row[0] for row in SARS]), axis=1)  # shape: (batch_size, 4)
            s = np.array([row[0] for row in SARS])  # shape: (batch_size, 4)
            # Fin Corrección

            # qvalue = self.model.Qpolicy.predict(s)[0]
            qvalue = self.model.Qpolicy.predict(s, verbose = 0)

            # Corrección
            # s_ = self.state_shape([row[3] for row in SARS])
            # s_ = np.squeeze(np.array([row[3] for row in SARS]), axis=1)  # shape: (batch_size, 4)
            s_ = np.array([row[3] for row in SARS])  # shape: (batch_size, 4)
            # Fin Corrección

            # future_qvalue = self.model.Qtarget.predict(s_)[0]
            future_qvalue = self.model.Qtarget.predict(s_, verbose = 0)

            for index, (state, action, reward, state_, done) in enumerate(SARS):
                if done == True:
                    Qtarget = reward
                else:
                    Qtarget = reward + self.gamma * np.max(future_qvalue[index])

                # Corrección
                qcurr = qvalue[index].copy()
                # qcurr = qvalue[index][0]
                # Fin Corrección
                qcurr[action] = Qtarget
                X.append(state)
                y.append(qcurr)

            # Corrección
            # X, y = np.array(X).reshape(1, self.batch_size, 1, self.states), np.array(y).reshape(1, self.batch_size, 1,
            #                                                                                     self.n_actions)
            X = np.array(X).reshape(self.batch_size, self.states)
            y = np.array(y).reshape(self.batch_size, self.n_actions)
            # Fin Corrección

            loss = self.model.Qpolicy.fit(X, y, batch_size=self.batch_size, shuffle=False, verbose=0)
            self.history.append(loss.history['loss'][0])

            if self.terminal_state:
                self.target_counter += 1

            # C -> target network update frequency
            if self.target_counter > self.C:
                self.model.Qtarget.set_weights(self.model.Qpolicy.get_weights())
                self.target_counter = 0

    def training(self):
        timestep_reward = 0

        for episode in tqdm(range(1, self.episodes + 1), ascii=True, unit='episode'):
            # s = self.env.reset()
            s, _ = self.env.reset()
            done = False
            score = 0
            while done != True:
                a = self.choose_action(s)
                s_, r, done, win, taca = self.env.step(a)

                # Update
                self.terminal_state = done
                self.update_D(s, a, r, s_, done)

                self.train()

                s = s_
                score += r

            self.decrement_epsilon(episode)

            # UPDATE
            self.reward_data.append(score)
            self.epsilon_data.append(self.epsilon)

            if episode % self.timestep == 0:
                self.graphs(episode)

        self.graphs(episode)
        self.model.Qpolicy.save(r'models/dqn_model.keras')

    def test_BORRAR(self, model_name, test_episodes = 100):
        model = tf.keras.models.load_model('{}'.format(model_name))
        reward = []
        self.epsilon = 0.05
        for i in range(test_episodes):
            ep_reward = 0
            s, _ = self.env.reset()
            done = False

            while done != True:
                if np.random.random() > self.epsilon:
                    # Corrección
                    s_, r, terminated, truncated, _ = self.env.step(a)
                    done = terminated or truncated
                    # s_, r, done, _ = self.env.step(a)
                    # Fin Corrección

                    s = s_
                    ep_reward += r

            reward.append(ep_reward)

        plt.plot([i for i in range(len(reward))], reward)
        plt.xlabel('Episodes')
        plt.ylabel('Episode reward')
        plt.savefig(r'img/Test_DQN_e{episode:03d}.png', dpi = 500)

    def test(self, test_episodes = 100, render=False):
        rewards = []
        self.epsilon = 0  # Desactiva exploración para usar sólo el modelo entrenado

        for i in range(test_episodes):
            ep_reward = 0
            state, _ = self.env.reset()
            done = False

            while not done:
                state_input = self.state_shape(state)
                action = np.argmax(self.model.Qpolicy.predict(state_input, verbose = 0))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                ep_reward += reward
                state = next_state

                if render:
                    self.env.render()

            print(f"🎯 Episodio {i + 1}: Recompensa total = {ep_reward}")
            rewards.append(ep_reward)

        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        print(f"\n📊 Recompensa promedio: {avg_reward:.1f}, Máxima: {max_reward}")

        # Guardar gráfica
        plt.plot(rewards)
        plt.xlabel('Episodios')
        plt.ylabel('Recompensa')
        plt.title('Evaluación del modelo DQN')
        plt.savefig('img/DQN_Test_result.png', dpi = 500)
        plt.close()


# Creación del agente ===========================
dqn = DQNAgent(episodes = 200)

# Entrenamiento ===========================
dqn.training()
