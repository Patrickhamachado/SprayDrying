import pandas as pd
import numpy as np
import tensorflow as tf
from cal_reward import RewardCalculator
import matplotlib.pyplot as plt
import random


def cargar_modelo(ruta_modelo='dqn_model.h5'):
    """Carga el modelo entrenado"""
    print(f"\nCargando modelo desde {ruta_modelo}...")
    return tf.keras.models.load_model(ruta_modelo)


def preparar_dato_test(datos_entrenamiento):
    """Selecciona y prepara una observación de los datos de entrenamiento"""
    print("\nPreparando dato de prueba desde datos de entrenamiento...")

    # Cargar datos originales
    datos = pd.read_csv(datos_entrenamiento)

    # Eliminar columna de tiempo si existe
    if 'time_stamp' in datos.columns:
        datos = datos.drop(columns=['time_stamp'])

    # Seleccionar una observación aleatoria
    random_idx = random.randint(0, len(datos) - 1)
    dato_test = datos.iloc[random_idx:random_idx + 1]

    print(f"\nObservación seleccionada (índice {random_idx}):")
    print(dato_test)

    return dato_test.values, dato_test


def predecir(modelo, dato):
    """Realiza predicción con el modelo"""
    print("\nRealizando predicción...")
    return modelo.predict(dato)


def evaluar_prediccion(dato_real, prediccion, calculator):
    """Evalúa y compara con el valor real"""
    print("\nEvaluando predicción...")

    # Calcular recompensa real
    reward_real = calculator.calculate_reward(dato_real.iloc[0])

    print("\n=== Resultados ===")
    print(f"Recompensa real calculada: {reward_real:.4f}")
    print(f"Recompensa predicha: {prediccion[0][0]:.4f}")  # Ajustar según tu modelo
    print(f"Diferencia: {abs(reward_real - prediccion[0][0]):.4f}")

    # Guardar resultados
    resultado = dato_real.copy()
    resultado['reward_real'] = reward_real
    resultado['reward_predicho'] = prediccion[0][0]

    return resultado


def main():
    try:
        # Paso 1: Cargar modelo
        modelo = cargar_modelo()

        # Paso 2: Preparar dato de prueba desde datos de entrenamiento
        dato_array, dato_df = preparar_dato_test('sample_data.csv')

        # Paso 3: Realizar predicción
        prediccion = predecir(modelo, dato_array)

        # Paso 4: Evaluar
        calculator = RewardCalculator("pesos.csv")
        resultado = evaluar_prediccion(dato_df, prediccion, calculator)

        # Paso 5: Mostrar y guardar resultados
        print("\nResumen completo:")
        print(resultado)
        resultado.to_csv('resultado_validacion.csv', index=False)
        print("\nResultado guardado en 'resultado_validacion.csv'")

    except Exception as e:
        print(f"\nError durante la validación: {str(e)}")
        raise


if __name__ == "__main__":
    main()
