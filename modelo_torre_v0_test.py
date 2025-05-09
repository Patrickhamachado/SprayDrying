
# script para probar la capacidad de predicción. Quizás haciendo 100 predicciones de periodos de 120 datos
# consecutivos (dos horas) cada uno y ver los desempeños de cada variable... Una imagen con la tendencia del
# promedio de cada una de las 100 épocas y otro mostrando un histograma o de densidad del valor real y predicho
# de cada variable, ya no del promedio de cada época sino incluyendo todos los datos, estaría muy bien!

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

# Rutas
DATASET_PATH = 'data/datos_Normal_a_e_s_7may.csv'
MODEL_PATH = 'models/modelo_torre.keras'
IMG_TENDENCIA = 'img/modelo_torre_eval_trends.png'
IMG_DENSIDAD = 'img/modelo_torre_eval_densidades.png'
IMG_ERRORES = 'img/modelo_torre_eval_errores.png'
WINDOW_SIZE = 120       # Tamaño de pasos de cada episodio

# Crear carpeta si no existe
os.makedirs('img', exist_ok=True)

# Cargar datos y modelo
df = pd.read_csv(DATASET_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo y datos cargados")

# Cargar los 5 modelos del entorno mejorado
model_paths = [f'models/modelo_torre_f{i}.keras' for i in range(1, 6)]
model_ensemble  = [tf.keras.models.load_model(path) for path in model_paths]

# Función para predecir usando ensamble
def ensemble_predict(input_data):
    predictions = [model(input_data, training=False).numpy() for model in model_ensemble]
    return np.mean(predictions, axis=0)


# Índices
idx_acciones = list(range(1, 9))
idx_externas = list(range(9, 19))
idx_estados = list(range(19, 35))  # s1–s16
input_cols = idx_acciones + idx_externas + idx_estados
output_cols = idx_estados

# Extraer datos como arrays
acciones = df.iloc[:, idx_acciones].values
externas = df.iloc[:, idx_externas].values
estados = df.iloc[:, idx_estados].values
resets = df['reset'].values

# Buscar inicios válidos (donde reset == 1) y donde haya al menos 120 pasos posteriores
# valid_starts = [i for i in range(len(df) - 120) if resets[i] == 1]
# valid_starts = valid_starts[:10]  # usar primeros 100 válido
valid_starts = [i for i in range(len(resets) - WINDOW_SIZE + 1) if np.all(resets[i:i + WINDOW_SIZE])]
# Elegir 100 al azar (o menos si hay menos disponibles)
random.seed(42)  # Para reproducibilidad
valid_starts = random.sample(valid_starts, min(100, len(valid_starts)))

predicciones = []
reales = []
epoca = 1

# Función para predecir ===================================
@tf.function
def predecir_estado(modelo, entrada):
    return modelo(entrada, training=False)

# Predicciones  ============================================
for start in valid_starts:
    print(f"Predicción {epoca}")
    estado_actual = estados[start].copy().astype(np.float32)
    pred_seq = []
    real_seq = []

    for t in range(WINDOW_SIZE):
        idx = start + t
        entrada_np = np.concatenate([acciones[idx], externas[idx], estado_actual])[None, :]  # forma (1, 34)
        entrada_tensor = tf.convert_to_tensor(entrada_np, dtype=tf.float32)

        estado_pred = predecir_estado(model, entrada_tensor)[0].numpy()
        # estado_pred = ensemble_predict(entrada_tensor)[0]

        pred_seq.append(estado_pred)
        real_seq.append(estados[idx + 1])

        estado_actual = estado_pred  # usar predicción como nuevo estado

    predicciones.append(np.array(pred_seq))
    reales.append(np.array(real_seq))
    epoca += 1

# Convertir a arrays
predicciones = np.array(predicciones)  # (100, 120, 16)
reales = np.array(reales)

# Nombres de variables
nombres_vars = [f's{i+1}' for i in range(16)]

# Aplanar datos
reales = reales.reshape(-1, reales.shape[-1])
predicciones = predicciones.reshape(-1, predicciones.shape[-1])

# ----- GRAFICO 1: MAE y MSE por variable -----
fig3, axes3 = plt.subplots(4, 4, figsize=(16, 12))
fig3.suptitle("Errores MAE y MSE por paso de tiempo (12,000 pasos)", fontsize=16)

for i in range(16):
    ax = axes3[i // 4, i % 4]

    real_vals = reales[:, i]
    pred_vals = predicciones[:, i]

    # Errores por paso
    mae_vals = np.abs(real_vals - pred_vals)
    mse_vals = (real_vals - pred_vals) ** 2

    # Graficar los 12,000 puntos de cada métrica
    ax.plot(mae_vals, label='MAE', color='skyblue', linewidth=0.8)
    ax.plot(mse_vals, label='MSE', color='salmon', linewidth=0.8)
    ax.set_title(f's{i + 1}')
    ax.set_xlim([0, len(real_vals)])
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(IMG_ERRORES)
plt.close()
print(f"Errores MAE y MSE guardados en {IMG_ERRORES}")


# ----- GRAFICO 2: Tendencia promedio por variable -----
fig1, axes1 = plt.subplots(4, 4, figsize=(16, 12))
fig1.suptitle("Tendencia promedio por variable (real vs. predicho)", fontsize=16)

for i in range(16):
    ax = axes1[i // 4, i % 4]
    ax.plot(reales[:, i], label='Real', linewidth=1)
    ax.plot(predicciones[:, i], '--', label='Predicho', linewidth=1)
    ax.set_title(f's{i + 1}')
    ax.set_xlim([0, len(reales)])

axes1[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(IMG_TENDENCIA)
plt.close()
print(f"Tendencias guardadas en {IMG_TENDENCIA}")


# ----- GRAFICO 3: Histogramas de todos los valores -----
fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12))
fig2.suptitle("Distribución de valores reales vs. predichos", fontsize=16)

for i in range(16):
    ax = axes2[i // 4, i % 4]
    ax.hist(reales[:, i], bins=50, alpha=0.6, label='Real', color='cornflowerblue')
    ax.hist(predicciones[:, i], bins=50, alpha=0.6, label='Predicho', color='sandybrown')
    ax.set_title(f's{i + 1}')


    # ax = axes2[i // 4, i % 4]
    # real_vals = reales[:, :, i].flatten()
    # pred_vals = predicciones[:, :, i].flatten()
    #
    # ax.hist(real_vals, bins=50, alpha=0.5, label='Real', density=True)
    # ax.hist(pred_vals, bins=50, alpha=0.5, label='Predicho', density=True)
    # ax.set_title(nombres_vars[i])
    # ax.grid(True)
    # if i == 0:
    #     ax.legend()
axes2[0, 0].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(IMG_DENSIDAD)
plt.close()
print(f"Densidades guardadas en {IMG_DENSIDAD}")



print("Fin exitoso")

#                   Conclusiones 7 mayo 5pm: ===================================

# De las tres gráficas:

# 1. Tendencias (Gráfica de líneas por variable)
# Aspectos positivos:
#
# Las predicciones ahora están acotadas entre 0 y 1 (¡bien hecho al usar sigmoid!).
#
# Para muchas variables (s3, s5, s6, s9, s10, s15, etc.), las tendencias generales son bastante razonables.
#
# Oportunidades de mejora:
#
# Hay cierta oscilación exagerada en algunas variables (s2, s4, s7), lo que sugiere que el modelo podría estar sobreajustando el corto plazo o teniendo problemas de acumulación de errores autoregresivos.
#
# En algunas variables (s11, s12, s13, s14), las predicciones están sistemáticamente desplazadas (bias).
#
# 🔹 2. MAE y MSE por paso de tiempo
# Bien:
#
# Los errores no se disparan fuera de control. Los picos son puntuales.
#
# El MAE se mantiene mayormente < 0.2 en casi todas las variables, lo cual es razonable si los datos están entre 0 y 1.
#
# Ajustes posibles:
#
# Muchos picos localizados indican zonas problemáticas. Tal vez valdría la pena analizar en qué condiciones del proceso ocurren para entender si se requiere una red más robusta o más datos representativos.
#
# Podrías suavizar visualmente con una media móvil si te interesa analizar tendencias más generales.
#
# 🔹 3. Histogramas de distribución real vs. predicha
# Muy útil:
#
# Aquí es donde mejor se aprecia el bias: hay variables como s2, s13, s14, s16 donde la distribución predicha está sistemáticamente desplazada respecto a la real.
#
# Otras (s6, s5, s10) se aproximan bastante bien.
#
# Ideas para refinar:
#
# Algunas distribuciones reales son multimodales (como s11 o s5) y la red predice más una sola moda. Puedes considerar arquitecturas con mayor capacidad o técnicas como mixture density networks (más avanzado).
#
# 🔚 Recomendaciones inmediatas
# Revisar el conjunto de datos: Si hay casos difíciles (picos de error), mirar si coinciden con ciertas condiciones externas o acciones.
#
# Reentrenar con más épocas o early stopping: El modelo parece estable, pero podrías afinar con más ciclos si usas validación con early stopping.
#
# Técnicas avanzadas (para más adelante):
#
# Incorporar dropout en entrenamiento y test (MC Dropout) para estimar incertidumbre.
#
# Probar LSTM si los estados tienen fuerte dependencia temporal.
#
# Revisar qué tan representados están los outliers del proceso.
