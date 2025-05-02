import pandas as pd
import tensorflow as tf
from cal_reward import RewardCalculator

# ========= CONFIGURACIÓN =========
DATA_PATH = 'data/datos_Normal_v2_26abr_V1Filter.csv'
MODEL_PATH = 'models/predict_model.keras'
PESOS_PATH = 'data/pesos.csv' # Assuming pesos.csv is in data directory
OPTIMIZATION_STEPS = 100 # Number of optimization steps
OPTIMIZATION_LEARNING_RATE = 0.01 # Learning rate for the optimizer

# Columnas que NO son predictores (variables de control/configuración) - Copied from predict_net.py
lista_acciones = ['Number_of_Jets_Open', 'Bombeo_Low_Pump_P_401', 'P404_High_Pump_Pressure_SP',
                   'Apertura_Valvula_Flujo_Aeroboost_FCV_0371', 'Apertura_Valvula_Presion_Aeroboost',
                   'Tower_Input_Air_Fan_Speed_Ref', 'Tower_Input_Temperature_SP', 'Tower_Internal_Pressure_SP']

# ========= CARGA DE DATOS Y DEFINICIÓN DE COLUMNAS =========
print("Cargando datos y definiendo columnas...")
try:
    # Load data just to get column names and structure
    data = pd.read_csv(DATA_PATH)
    if 'time_stamp' in data.columns:
        data = data.drop(columns=['time_stamp'])

    list_cols = data.columns.tolist()
    lista_estados = [col for col in list_cols if col not in lista_acciones]

    print(f"Total de características (input columns): {len(list_cols)}")
    print(f"Variables a predecir (output columns): {len(lista_estados)}")
    print("Columnas de entrada:", list_cols[:5], "...")
    print("Columnas de salida (predichas):", lista_estados[:5], "...")

except FileNotFoundError:
    print(f"Error: El archivo de datos no se encontró en {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error al cargar o procesar el archivo de datos: {e}")
    exit()

# ========= DEFINICIÓN DE VARIABLES PARA D-RTO =========
print("\nDefiniendo variables de decisión y entradas fijas para D-RTO...")
list_decision_vars = lista_acciones # Variables a optimizar
list_fixed_inputs = [col for col in list_cols if col not in list_decision_vars] # Resto de las entradas al modelo

print(f"Variables de Decisión ({len(list_decision_vars)}): {list_decision_vars}")
print(f"Entradas Fijas ({len(list_fixed_inputs)}): {list_fixed_inputs[:5]} ...")


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

# ========= CARGA DEL CALCULADOR DE RECOMPENSA =========
print(f"\nCargando calculador de recompensa desde {PESOS_PATH}...")
try:
    reward_calculator = RewardCalculator(PESOS_PATH)
    print("Columnas requeridas por el calculador de recompensa:", reward_calculator.required_columns)

    # Verify that all required reward columns are in the predicted outputs
    missing_reward_cols = [col for col in reward_calculator.required_columns if col not in lista_estados]
    if missing_reward_cols:
        print(f"ADVERTENCIA: Las siguientes columnas requeridas por el calculador de recompensa no están en las variables predichas: {missing_reward_cols}")
        print("La recompensa calculada solo considerará las columnas predichas disponibles.")
    else:
        print("Todas las columnas requeridas por el calculador de recompensa están en las variables predichas.")

except FileNotFoundError:
    print(f"Error: El archivo de pesos no se encontró en {PESOS_PATH}")
    exit()
except Exception as e:
    print(f"Error al cargar el calculador de recompensa: {e}")
    exit()

# Get reward weights and map them to predicted output indices for TensorFlow calculation
reward_weights = reward_calculator.pesos_dict
base_var = 'Torre_PB_Flujo_Schenck'
base_weight = reward_weights.get(base_var, 0.0) # Get base weight, default to 0 if not found

# Create lists to store weights and corresponding predicted output indices for TensorFlow calculation
ordered_reward_vars = []
ordered_reward_weights = []
ordered_reward_indices = []
base_idx_in_predicted = -1 # Index of the base variable in predicted_state_tensor (within list_PREDICT order)

# Find indices of reward variables in list_PREDICT and ensure order matches weights
predicted_col_to_idx = {col: i for i, col in enumerate(lista_estados)}

for var, weight in reward_weights.items():
     if var in predicted_col_to_idx: # Check if this reward variable is one of the predicted outputs
         ordered_reward_vars.append(var)
         ordered_reward_weights.append(weight)
         ordered_reward_indices.append(predicted_col_to_idx[var])
         if var == base_var:
             base_idx_in_predicted = predicted_col_to_idx[var]


if not ordered_reward_vars:
     print("WARNING: None of the reward variables are in the predicted outputs (list_PREDICT). Differentiable reward will be zero.")
     # Handle case where no reward variables are predicted
     tf_ordered_reward_indices = tf.constant([], dtype=tf.int32)
     tf_ordered_reward_weights = tf.constant([], dtype=tf.float32)
     base_idx_in_predicted = -1 # Ensure this remains -1
else:
    tf_ordered_reward_indices = tf.constant(ordered_reward_indices, dtype=tf.int32)
    tf_ordered_reward_weights = tf.constant(ordered_reward_weights, dtype=tf.float32)


# ========= FUNCIÓN PARA CALCULAR RECOMPENSA DE LA PREDICCIÓN (USANDO CALCULADORA ORIGINAL) =========

# Ensure the order of predicted columns matches the expected order if needed by the reward calculator
# The model output is a tensor, the reward calculator expects a pandas Series/DataFrame row
# We need to map the tensor output to a Series with correct column names

def calculate_predicted_reward(input_data_row_tensor):
    """
    Takes a single row of input data as a TensorFlow tensor,
    predicts the next state using the model, and calculates the reward
    for the predicted state using the original RewardCalculator.
    """
    # Ensure input_data_row_tensor is the correct shape (1, len(list_cols))
    # This check might need adjustment if batch processing is used later
    if input_data_row_tensor.shape != (1, len(list_cols)):
         # Attempt to reshape if it's a flat tensor/array, assuming a single sample
         if input_data_row_tensor.ndim == 1 and input_data_row_tensor.shape[0] == len(list_cols):
             input_data_row_tensor = tf.expand_dims(input_data_row_tensor, axis=0)
         else:
             raise ValueError(f"Input tensor shape {input_data_row_tensor.shape} is not compatible with expected shape (1, {len(list_cols)}))")


    # Predict the next state
    predicted_state_tensor = model(input_data_row_tensor)

    # Convert the predicted state tensor to a pandas Series for the reward calculator
    # Ensure the order of columns in the Series matches list_PREDICT
    predicted_state_series = pd.Series(predicted_state_tensor.numpy().flatten(), index=lista_estados)

    # Calculate the reward for the predicted state
    # The reward_calculator expects a DataFrame row (axis=1 in apply) or a Series
    # Passing a single Series is equivalent to a DataFrame with one row
    try:
        predicted_reward = reward_calculator.calculate_reward(predicted_state_series)
    except ValueError as ve:
        print(f"Error calculating reward: {ve}")
        # Handle cases where reward calculation fails, perhaps return a very low reward
        predicted_reward = -1e9 # Assign a very low reward
    except Exception as e:
        print(f"An unexpected error occurred during reward calculation: {e}")
        predicted_reward = -1e9


    return predicted_reward, predicted_state_series


# ========= FUNCIÓN PARA CALCULAR RECOMPENSA DIFERENCIABLE (TENSORFLOW) =========
def calculate_differentiable_reward(predicted_state_tensor, reward_weights_tf, reward_indices_tf, base_var_index_in_predicted, base_weight):
    """
    Calculates the reward using TensorFlow operations, allowing for gradients.

    Args:
        predicted_state_tensor (tf.Tensor): Tensor of predicted state variables (shape (1, len(list_PREDICT))).
        reward_weights_tf (tf.Constant): Tensor of reward weights in the order of reward_indices_tf.
        reward_indices_tf (tf.Constant): Tensor of indices of reward variables within predicted_state_tensor.
        base_var_index_in_predicted (int): Index of the base variable ('Torre_PB_Flujo_Schenck') within list_PREDICT.
        base_weight (float): Weight of the base variable.

    Returns:
        tf.Tensor: The calculated reward (shape (1,)).
    """

    if tf.size(reward_indices_tf) == 0:
        return tf.constant([0.0], dtype=tf.float32) # Return 0 if no reward variables are predicted

    # Extract the predicted states for the reward variables using gathered indices
    # predicted_reward_states_tensor = tf.gather(predicted_state_tensor, reward_indices_tf, axis=1)
    # Note: tf.gather is tricky with gradients on the index. Element-wise multiplication and reduction is safer if order is aligned.

    # Ensure predicted_state_tensor is squeezed to (len(list_PREDICT),) if it's (1, len(list_PREDICT))
    # Or handle batch dimension consistently.
    # Let's assume predicted_state_tensor is (1, len(list_PREDICT))

    # Extract the predicted states for the reward variables in the defined order (matching weights/indices)
    predicted_reward_states_ordered = tf.gather(predicted_state_tensor, reward_indices_tf, axis=1) # Shape (1, num_reward_vars)

    # Perform weighted sum: sum(value * weight) for all reward variables
    weighted_predicted_terms = predicted_reward_states_ordered * reward_weights_tf # Shape (1, num_reward_vars)

    # Sum up all weighted terms
    total_weighted_sum = tf.reduce_sum(weighted_predicted_terms, axis=1) # Shape (1,)

    # The original logic is base_value * base_weight - sum(other_values * other_weights)
    # Let's find the base term and the sum of negative terms separately.

    # Find the base term value
    # Need the index of the base variable within the *ordered* reward variables
    try:
        base_var_order_idx = ordered_reward_vars.index(base_var) # Use the Python list for finding index
    except ValueError:
        print(f"Error: Base variable '{base_var}' not found in ordered_reward_vars. Cannot calculate differentiable reward.")
        return tf.constant([-1e9], dtype=tf.float32) # Return a very low reward on error

    predicted_base_value = predicted_reward_states_ordered[0, base_var_order_idx] # Shape () scalar
    base_term_tf = predicted_base_value * tf.constant(base_weight, dtype=tf.float32) # Shape () scalar
    base_term_tf = tf.expand_dims(base_term_tf, axis=0) # Shape (1,)

    # Calculate the sum of weighted negative components (all except the base)
    negative_terms_mask = [var != base_var for var in ordered_reward_vars]
    negative_terms_weights_tf = tf.boolean_mask(tf_ordered_reward_weights, negative_terms_mask)
    predicted_negative_states_ordered = tf.boolean_mask(predicted_reward_states_ordered, negative_terms_mask, axis=1) # Shape (1, num_negative_vars)

    if tf.size(negative_terms_weights_tf) > 0:
         sum_weighted_negatives = tf.reduce_sum(predicted_negative_states_ordered * negative_terms_weights_tf, axis=1) # Shape (1,)
    else:
         sum_weighted_negatives = tf.constant([0.0], dtype=tf.float32) # Shape (1,)

    # Calculate the differentiable predicted reward
    predicted_reward_tf = base_term_tf - sum_weighted_negatives # Shape (1,)

    return predicted_reward_tf


# ========= D-RTO OPTIMIZATION IMPLEMENTATION =========

def optimize_inputs(current_state_series, initial_decision_vars_guess):
    """
    Optimizes the decision variables to maximize the predicted reward
    for the next time step, given the current state.

    Args:
        current_state_series (pd.Series): A pandas Series representing the current
                                          state of all relevant process variables (list_cols).
        initial_decision_vars_guess (dict): A dictionary with initial guess values
                                            for the decision variables (list_decision_vars).

    Returns:
        dict: A dictionary containing the optimized decision variable values.
        float: The predicted reward at the optimized inputs (calculated using the original function for accuracy).
        pd.Series: The predicted next state at the optimized inputs (calculated using the original function).
    """
    print("\nStarting D-RTO optimization...")

    # Separate fixed inputs from the current state and convert to TensorFlow constant
    fixed_input_values_tf = tf.constant([current_state_series[col] for col in list_fixed_inputs], dtype=tf.float32)

    # Initialize decision variables as TensorFlow variables
    # Use the provided initial guess
    initial_guess_tensor = tf.constant([initial_decision_vars_guess[var] for var in list_decision_vars], dtype=tf.float32)
    decision_vars_tf = tf.Variable(initial_guess_tensor, dtype=tf.float32)

    # Use an optimizer (Adam is a good general choice)
    optimizer = tf.optimizers.Adam(learning_rate=OPTIMIZATION_LEARNING_RATE)

    print(f"Optimizing for {OPTIMIZATION_STEPS} steps...")

    for step in range(OPTIMIZATION_STEPS):
        with tf.GradientTape() as tape:
            # Ensure tape is watching the decision variables
            tape.watch(decision_vars_tf)

            # Combine decision variables and fixed inputs in the correct order of list_cols
            # Create a list of tensors/values in the correct order of list_cols
            ordered_input_values_tensors = []
            fixed_input_idx = 0
            decision_var_idx = 0
            for col in list_cols:
                 if col in list_fixed_inputs:
                     ordered_input_values_tensors.append(fixed_input_values_tf[fixed_input_idx])
                     fixed_input_idx += 1
                 elif col in list_decision_vars:
                     ordered_input_values_tensors.append(decision_vars_tf[decision_var_idx])
                     decision_var_idx += 1

            # Stack the ordered values into a single tensor (shape: 1, len(list_cols))
            input_tensor_for_model = tf.stack(ordered_input_values_tensors)
            input_tensor_for_model = tf.expand_dims(input_tensor_for_model, axis=0) # Add batch dimension


            # Predict the next state using the model (inside tape)
            predicted_state_tensor = model(input_tensor_for_model) # Shape (1, len(list_PREDICT))

            # Calculate the differentiable predicted reward (inside tape)
            predicted_reward_tf = calculate_differentiable_reward(predicted_state_tensor,
                                                                  tf_ordered_reward_weights,
                                                                  tf_ordered_reward_indices,
                                                                  base_idx_in_predicted,
                                                                  base_weight)

            # The objective is to maximize reward, so we minimize the negative reward
            objective_loss = -predicted_reward_tf # Shape (1,)

        # Compute gradients OUTSIDE the tape
        gradients = tape.gradient(objective_loss, decision_vars_tf)

        # Apply gradients
        # Ensure gradients are not None (can happen if something went wrong in tape or no reward vars are predicted)
        if gradients is not None:
             optimizer.apply_gradients(zip([gradients], [decision_vars_tf]))
        else:
             print(f"Warning: No gradients computed at step {step+1}.")

        # Optional: Print progress (need to convert objective_loss back to numpy here)
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{OPTIMIZATION_STEPS}, Predicted Reward: {-objective_loss.numpy()[0]:.4f}") # Print actual reward from differentiable calc

    # Optimization finished. Get the optimized decision variables (numpy)
    optimized_decision_vars_tensor = decision_vars_tf.numpy()

    # Reconstruct the full input tensor with optimized decision variables (using numpy for final evaluation)
    final_input_values_np = []
    decision_vars_optimized_dict = dict(zip(list_decision_vars, optimized_decision_vars_tensor))

    fixed_input_values_np = current_state_series[list_fixed_inputs].values # Get fixed inputs as numpy
    fixed_input_idx = 0
    decision_var_idx = 0
    for col in list_cols:
         if col in list_fixed_inputs:
             final_input_values_np.append(fixed_input_values_np[fixed_input_idx])
             fixed_input_idx += 1
         elif col in list_decision_vars:
             final_input_values_np.append(decision_vars_optimized_dict[col])
             decision_var_idx += 1

    final_input_tensor_for_eval = tf.constant([final_input_values_np], dtype=tf.float32)


    # Calculate the final predicted state and reward using the original function (outside tape) for reporting
    # This ensures the reported values match the original reward calculation logic.
    final_predicted_reward_eval, final_predicted_state_series = calculate_predicted_reward(final_input_tensor_for_eval)

    print("\nOptimization complete.")
    print(f"Optimized Predicted Reward (evaluated with original function): {final_predicted_reward_eval:.4f}")
    print("Optimized Decision Variables:")
    # Convert optimized tensor back to dict for clear output
    optimized_decision_vars_dict = dict(zip(list_decision_vars, optimized_decision_vars_tensor))
    for var, val in optimized_decision_vars_dict.items():
        print(f"  {var}: {val:.4f}")
    print("Predicted State with Optimized Variables:")
    print(final_predicted_state_series.to_string())

    return optimized_decision_vars_dict, final_predicted_reward_eval, final_predicted_state_series


# ========= EJEMPLO DE USO DEL OPTIMIZADOR D-RTO =========
print("\n=== Demostración del Optimizador D-RTO ===")

if not data.empty:
    # Use the first row of the original data as the current state for the optimizer
    current_state_example = data.iloc[0] # This is a pandas Series

    print("\nEstado actual de ejemplo (primera fila del dataset):")
    print(current_state_example.to_string())

    # Provide an initial guess for the decision variables
    # Using the actual values from the current state as an initial guess
    initial_decision_vars_guess = current_state_example[list_decision_vars].to_dict()

    print("\nIntial guess for Decision Variables:")
    for var, val in initial_decision_vars_guess.items():
         print(f"  {var}: {val:.4f}")


    # Run the optimization
    optimized_decision_vars, predicted_reward_at_optimum, predicted_state_at_optimum = optimize_inputs(current_state_example,
                                                                                                       initial_decision_vars_guess)

    print("\nResultados de la Optimización (Evaluado con la función original de recompensa):")
    print("Variables de Decisión Óptimas:")
    for var, val in optimized_decision_vars.items():
         print(f"  {var}: {val:.4f}")
    print(f"Recompensa Predicha con Variables Óptimas: {predicted_reward_at_optimum:.4f}")
    print("Estado Predicho con Variables Óptimas:")
    # print(predicted_state_at_optimum.to_string())


else:
    print("El dataset está vacío, no se puede ejecutar la demostración del optimizador.")


print("\nProceso completado!")
