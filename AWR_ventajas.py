# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import os
#
# # ==== CONFIGURACIÓN ====
# CSV_PATH = "data/datos_Normal_a_e_s_7may.csv"
# CRITIC_PATH = "models/warmup_critic.keras"
# IMG_DIR = "img"
# os.makedirs(IMG_DIR, exist_ok=True)
#
# ADV_BETAS = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
#
# # ==== CARGAR DATOS ====
# df = pd.read_csv(CSV_PATH)
# df = df[df['reset'] == 1].reset_index(drop=True)
#
# state_cols = [f"s{i+1}" for i in range(16)]
# external_cols = [f"e{i+1}" for i in range(10)]
# action_cols = [f"a{i+1}" for i in range(8)]
#
# states = df[state_cols].values.astype(np.float32)
# externals = df[external_cols].values.astype(np.float32)
# actions = df[action_cols].values.astype(np.float32)
# inputs = np.concatenate([externals, states], axis=1)
#
# # ==== CARGAR CRÍTICO ====
# critic = tf.keras.models.load_model(CRITIC_PATH)
#
# # ==== CALCULAR VENTAJAS ====
# q_vals = critic.predict([inputs, actions], verbose=1).squeeze()
# baseline = np.mean(q_vals)
# advantages = q_vals - baseline
#
# # ==== GRÁFICA DE PESOS PARA DIFERENTES BETA ====
# plt.figure(figsize=(12, 6))
# for beta in ADV_BETAS:
#     weights = np.exp(advantages / beta)
#     weights = np.clip(weights, 0, 20)  # evitar colas infinitas en la visualización
#     plt.hist(weights, bins=100, alpha=0.6, label=f"β = {beta}", histtype='step')
#
# plt.title("Distribución de pesos exp(adv / β)")
# plt.xlabel("Peso")
# plt.ylabel("Frecuencia")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(f"{IMG_DIR}/adv_weights_by_beta.png")
# plt.close()
#
# print("✅ Distribución de pesos guardada en img/adv_weights_by_beta.png")
#
#

# ======================================================

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import os
#
# # ==== CONFIGURACIÓN ====
# CSV_PATH = "data/datos_Normal_a_e_s_7may.csv"
# CRITIC_PATH = "models/warmup_critic.keras"
#
# # ==== CARGAR DATOS ====
# df = pd.read_csv(CSV_PATH)
# df = df[df['reset'] == 1].reset_index(drop=True)
#
# state_cols = [f"s{i+1}" for i in range(16)]
# external_cols = [f"e{i+1}" for i in range(10)]
# action_cols = [f"a{i+1}" for i in range(8)]
#
# states = df[state_cols].values.astype(np.float32)
# externals = df[external_cols].values.astype(np.float32)
# actions = df[action_cols].values.astype(np.float32)
#
# inputs = np.concatenate([externals, states], axis=1)
#
# # ==== CARGAR CRÍTICO ====
# critic = tf.keras.models.load_model(CRITIC_PATH)
#
# # ==== CALCULAR Q(s, a) y V(s) ====
# q_values = critic.predict([inputs, actions], verbose=1).squeeze()
# baseline = np.mean(q_values)
# advantages = q_values - baseline
#
# # ==== ESTADÍSTICAS ====
# print(f"📊 Ventajas (Q(s,a) - baseline):")
# print(f"  Media:      {advantages.mean():.5f}")
# print(f"  Desv. std.: {advantages.std():.5f}")
# print(f"  Máximo:     {advantages.max():.5f}")
# print(f"  Mínimo:     {advantages.min():.5f}")
#
# # ==== HISTOGRAMA ====
# plt.figure(figsize=(10, 6))
# plt.hist(advantages, bins=100, color='skyblue', edgecolor='k', alpha=0.7)
# plt.title("Distribución de ventajas (Q(s,a) - baseline)")
# plt.xlabel("Ventaja")
# plt.ylabel("Frecuencia")
# plt.grid(True)
# plt.axvline(0, color='red', linestyle='--', label="Línea base (0)")
# plt.legend()
# plt.tight_layout()
# plt.savefig("img/ventajas_distribution.png")
# plt.close()
#
# print("✅ Visualización guardada en img/ventajas_distribution.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# ==== CONFIGURACIÓN ====
CSV_PATH = "data/datos_Normal_a_e_s_7may.csv"
CRITIC_PATH = "models/warmup_critic.keras"
ADV_BETAS = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
CLIP_VALUE = 20

# ==== CARGA DE DATOS ====
df = pd.read_csv(CSV_PATH)
df = df[df['reset'] == 1].reset_index(drop=True)

state_cols = [f"s{i+1}" for i in range(16)]
external_cols = [f"e{i+1}" for i in range(10)]
action_cols = [f"a{i+1}" for i in range(8)]

states = df[state_cols].values.astype(np.float32)
externals = df[external_cols].values.astype(np.float32)
actions = df[action_cols].values.astype(np.float32)
inputs = np.concatenate([externals, states], axis=1)

# ==== CARGA DE MODELO CRÍTICO ====
critic = tf.keras.models.load_model(CRITIC_PATH)

# ==== CÁLCULO DE VENTAJAS ====
q_vals = critic.predict([inputs, actions], verbose=1).squeeze()
baseline = np.mean(q_vals)
advantages = q_vals - baseline

# ==== GRAFICAR DISTRIBUCIÓN DE VENTAJAS ====
plt.figure(figsize=(10, 6))
plt.hist(advantages, bins=80, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Línea base (0)')
plt.title("Distribución de ventajas (Q(s,a) - baseline)")
plt.xlabel("Ventaja")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("img/ventajas_distribution.png")
plt.close()

# ==== VARIANTES DE CÁLCULO DE PESOS ====
weights_exp = {}
weights_clip = {}
weights_norm = {}
weights_relu = np.maximum(advantages, 0)

for beta in ADV_BETAS:
    w = np.exp(advantages / beta)
    weights_exp[beta] = w
    weights_clip[beta] = np.clip(w, 0, CLIP_VALUE)
    weights_norm[beta] = w / np.max(w)  # Normalización simple

# ==== GRAFICAR PESOS - EXP(adv / β) SIN CLIP ====
plt.figure(figsize=(12, 6))
for beta in ADV_BETAS:
    plt.hist(weights_exp[beta], bins=80, alpha=0.6, label=f"β = {beta}", histtype='step')
plt.title("Distribución de pesos exp(adv / β)")
plt.xlabel("Peso")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("img/adv_weights_by_beta.png")
plt.close()

# ==== GRAFICAR PESOS - CLIP ====
plt.figure(figsize=(12, 6))
for beta in ADV_BETAS:
    plt.hist(weights_clip[beta], bins=80, alpha=0.6, label=f"β = {beta}", histtype='step')
plt.title(f"Distribución de pesos exp(adv / β) con clip={CLIP_VALUE}")
plt.xlabel("Peso")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("img/adv_weights_clip_by_beta.png")
plt.close()

# ==== GRAFICAR PESOS - NORMALIZADOS ====
plt.figure(figsize=(12, 6))
for beta in ADV_BETAS:
    plt.hist(weights_norm[beta], bins=80, alpha=0.6, label=f"β = {beta}", histtype='step')
plt.title("Distribución de pesos exp(adv / β) normalizados [0,1]")
plt.xlabel("Peso normalizado")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("img/adv_weights_normalized_by_beta.png")
plt.close()

# ==== GRAFICAR RE-LU DE VENTAJAS ====
plt.figure(figsize=(10, 6))
plt.hist(weights_relu, bins=80, color='orange', edgecolor='black')
plt.title("Distribución de pesos ReLU(ventaja) = max(0, Q(s,a) - baseline)")
plt.xlabel("Peso")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig("img/adv_weights_relu.png")
plt.close()

print("✅ Gráficas de distribución de pesos generadas.")
