import pandas as pd
import tensorflow as tf
import json # Importar la librería json

# ========= CONFIGURACIÓN =========
DATA_PATH = 'data/datos_Normal_v2_26abr_V1Filter.csv'
MODEL_PATH = 'models/predict_model.keras'
OUTPUT_JSON_PATH = 'data/sample_prediction.json' # Ruta para el archivo JSON

# Columnas que NO son predictores (variables de control/configuración)
list_no_predict = ['Number_of_Jets_Open', 'Bombeo_Low_Pump_P_401', 'P404_High_Pump_Pressure_SP',
                   'Apertura_Valvula_Flujo_Aeroboost_FCV_0371', 'Apertura_Valvula_Presion_Aeroboost',
                   'Tower_Input_Air_Fan_Speed_Ref', 'Tower_Input_Temperature_SP', 'Tower_Internal_Pressure_SP']

# ========= CARGA DE DATOS Y DEFINICIÓN DE COLUMNAS =========
print("Cargando datos y definiendo columnas...")
try:
    data = pd.read_csv(DATA_PATH)
    # Eliminar columna temporal si existe
    if 'time_stamp' in data.columns:
        data = data.drop(columns=['time_stamp'])

    list_cols = data.columns.tolist()
    list_PREDICT = [col for col in list_cols if col not in list_no_predict]

    print(f"Total de características (input columns): {len(list_cols)}")
    print(f"Variables a predecir (output columns): {len(list_PREDICT)}")
    print("Columnas de entrada:", list_cols[:5], "...")
    print("Columnas de salida (predichas):", list_PREDICT[:5], "...")

except FileNotFoundError:
    print(f"Error: El archivo de datos no se encontró en {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error al cargar o procesar el archivo de datos: {e}")
    exit()

# ========= CARGA DEL MODELO =========
print(f"\nCargando modelo desde {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
except FileNotFoundError:
    print(f"Error: El archivo del modelo no se encontró en {MODEL_PATH}")
    print("Asegúrate de haber ejecutado predict_net.py primero para entrenar y guardar el modelo.")
    exit()
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# ========= SELECCIONAR MUESTRA ALEATORIA =========
print("\nSeleccionando una muestra aleatoria del dataset...")
if not data.empty:
    # Usamos un random_state fijo para reproducibilidad si lo deseas
    random_sample = data.sample(n=1, random_state=42)
    input_data = random_sample[list_cols].values # Extract values for input columns
    # Obtener los valores reales para las columnas a predecir como una lista
    real_values = random_sample[list_PREDICT].values[0].tolist()

    print("\nDatos de entrada para la predicción (muestra aleatoria):")
    print(random_sample[list_cols])
    print("\nValores reales para las columnas a predecir en esta muestra:")
    # Imprimir los valores reales con los nombres de columna
    real_print_dict = dict(zip(list_PREDICT, real_values))
    for col, value in real_print_dict.items():
         print(f"  {col}: {value:.4f}")

else:
    print("Error: El dataset está vacío.")
    exit()


# ========= REALIZAR PREDICCIÓN =========
print("\nRealizando predicción...")
try:
    prediction = model.predict(input_data)
    # Obtener los valores predichos como una lista
    predicted_values = prediction[0].tolist()
    print("\nPredicción generada:")

    # ========= PREPARAR DATOS PARA JSON EN EL NUEVO FORMATO =========
    output_data = {}
    for i, col_name in enumerate(list_PREDICT):
        output_data[col_name] = {"Real": real_values[i],
                                 "Predict": predicted_values[i]}

    # Imprimir predicción y real con nombres de columna (usando el nuevo diccionario)
    print("\nComparación de valores Real vs Predicción:")
    for col, values in output_data.items():
        print(f"  {col}: Real={values['Real']:.4f}, Predict={values['Predict']:.4f}")


    # ========= GUARDAR EN JSON =========
    print(f"\nGuardando valores reales y predichos en {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print("Archivo JSON guardado exitosamente.")


except Exception as e:
    print(f"Error al realizar la predicción o guardar el JSON: {e}")

print("\nProceso completado!")
