import matplotlib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
matplotlib.use('Agg')  # Configuración para entornos no interactivos
import matplotlib.pyplot as plt

# predict_net_v2.py

#  Script para entrenar un modelo de red neuronal que actúa como modelo dinámico del proceso. Este modelo
#  (models/predict_model.keras) predice el siguiente estado del sistema dadas las entradas actuales y es
#  utilizado por el optimizador D-RTO.

# 29 abr 2025: Corregido que estab prediciendo los mismos datos de entrada, en lugar de los del siguiente minuto


# ========= CONFIGURACIÓN =========
EPOCHS = 4
VALIDATION_SPLIT = 0.1
TEST_SIZE = 0.1
RANDOM_STATE = 137
LEARNING_RATE = 0.0001

# ========= CARGA DE DATOS =========
print("Cargando datos...")
data = pd.read_csv('data/datos_Normal_v2_26abr_V1Filter.csv')

# Verificación de columnas
print("\nColumnas disponibles en el dataset:")
print(data.columns.tolist())

# Eliminar columna temporal si existe
if 'time_stamp' in data.columns:
    data = data.drop(columns=['time_stamp'])

# ========= DEFINICIÓN DE COLUMNAS =========
print("\nDefiniendo columnas...")
list_cols = data.columns.tolist()

# Columnas que NO son predictores (variables de control/configuración)
list_no_predict = ['Number_of_Jets_Open', 'Bombeo_Low_Pump_P_401', 'P404_High_Pump_Pressure_SP',
                   'Apertura_Valvula_Flujo_Aeroboost_FCV_0371', 'Apertura_Valvula_Presion_Aeroboost',
                   'Tower_Input_Air_Fan_Speed_Ref', 'Tower_Input_Temperature_SP', 'Tower_Internal_Pressure_SP']

# Columnas a predecir
list_PREDICT = [col for col in list_cols if col not in list_no_predict]

print(f"\nTotal de características: {len(list_cols)}")
print(f"Variables a predecir: {len(list_PREDICT)}")
print("Ejemplo de variables a predecir:", list_PREDICT[:5])

# ========= DIVISIÓN DE DATOS =========
print("\nDividiendo datos...")
train_data = data.iloc[0:-1].sample(frac=1-TEST_SIZE, random_state=RANDOM_STATE)
test_data = data.iloc[0:-1].drop(train_data.index)

# Los valores Y a predecir, son la siguiente fila de los X de entrada
train_data_y = data.iloc[train_data.index + 1]
train_data_y = train_data_y[list_PREDICT]

test_data_y = data.iloc[test_data.index + 1]
test_data_y = test_data_y[list_PREDICT]

print(f"Tamaño del conjunto de entrenamiento: {len(train_data)}")
print(f"Tamaño del conjunto de prueba: {len(test_data)}")

# ========= CONSTRUCCIÓN DEL MODELO =========
print("\nConstruyendo modelo...")
model = Sequential([Input(shape=(len(list_cols),)),
                    Dense(64, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(len(list_PREDICT))])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mse',
              metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])

model.summary()

# ========= ENTRENAMIENTO =========
print("\nIniciando entrenamiento...")
history = model.fit(train_data[list_cols],
                    train_data_y,
                    epochs=EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1)

# ========= EVALUACIÓN =========
print("\nEvaluando modelo...")
test_results = model.evaluate(test_data[list_cols], test_data_y, verbose=0)

# Predicciones para métricas adicionales
y_pred = model.predict(test_data[list_cols])
y_true = test_data[list_PREDICT].values

print("\n=== MÉTRICAS PRINCIPALES ===")
print(f"Test Loss (MSE): {test_results[0]:.4f}")
print(f"Test MAE: {test_results[1]:.4f}")
print(f"Test RMSE: {test_results[2]:.4f}")

print("\n=== MÉTRICAS ADICIONALES ===")
print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²: {r2_score(y_true, y_pred):.4f}")

# ========= PREDICCIÓN DE EJEMPLO =========
print("\nPreparando ejemplo de predicción...")
# Usamos la primera fila del test set como ejemplo
vals = test_data[list_cols].iloc[0:1].values

print("\nDatos de entrada para la predicción (primera fila de test):")
print(vals)

prediction = model.predict(vals)
print("\nResultado de la predicción:")
print(prediction)

# ========= VISUALIZACIÓN =========
print("\nGenerando gráficas...")
plt.figure(figsize=(15, 6))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Evolución de la Pérdida (MSE)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Gráfico de MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Evolución del Error Absoluto Medio (MAE)')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(f'img/training_metrics.png')
print("\nGráficas guardadas en training_metrics.png")

# ========= GUARDAR MODELO =========
model.save('models/predict_model.keras')
print("\nModelo guardado como spray_drying_model.keras")

print("\nProceso completado exitosamente!")
