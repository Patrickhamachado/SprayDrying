import pandas as pd
import tensorflow as tf
from cal_reward import RewardCalculator
import random

def cargar_modelo(ruta_modelo='dqn_model.h5'):
    """Carga el modelo usando TensorFlow directamente"""
    print(f"\nCargando modelo desde {ruta_modelo}...")
    return tf.keras.models.load_model(ruta_modelo,
                                      custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

def preparar_dato_test(datos_entrenamiento='datos_Normal_v2_26abr_V1Filter.csv'):
    """Versión simplificada y robusta"""
    print("\nPreparando dato de prueba...")

    # Cargar y validar datos
    datos = pd.read_csv(datos_entrenamiento)
    datos = datos.drop(columns=['time_stamp'], errors='ignore')

    # Selección segura
    random_idx = random.randint(0, len(datos)-1)
    return datos.iloc[random_idx:random_idx+1].values, datos.iloc[random_idx:random_idx+1]

def main():
    try:
        # Cargar modelo
        modelo = cargar_modelo()

        # Preparar dato
        dato_array, dato_df = preparar_dato_test()

        # Predecir
        prediccion = modelo.predict(dato_array)

        # Calcular recompensa real
        calculator = RewardCalculator("pesos.csv")
        reward_real = calculator.calculate_reward(dato_df.iloc[0])

        # Resultados
        print("\n=== Resultado de Validación ===")
        print(f"Recompensa Real: {reward_real:.4f}")
        print(f"Recompensa Predicha: {prediccion[0][0]:.4f}")
        print(f"Diferencia: {abs(reward_real - prediccion[0][0]):.4f}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
