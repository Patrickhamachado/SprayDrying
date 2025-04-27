# Proyecto de Optimización de Secado con DQN

Sistema de aprendizaje por refuerzo profundo para optimización del proceso de secado por aspersión.

## Estructura del Proyecto

Se generan dos redes neuronales:
- **Red_simulacion**: Simula la torre de secado. Las entradas son el conjunto de estados (26 variables: 10 de Perturbación y 16 Estados) y acciones (8 variables), para un total de 34 variables y la salida es el estado en el minuto siguiente (26 variables)  

- **Red_optimizacion_secado**: Esta red, construye la politica para optimizar económicamente la operación de la torre de secado. Contará con 26 variables de entrada, los estados y como salida calcula las 8 variables de acción que constituyen la mejor opción económicamente. Para interactuar y aprender del Entorno, utiliza la primera red neuronal.


## Descripción de las variables

### Variables de perturbación

Las diez variables marcadas como "Perturbación" en la hoja Definitivos_35 del excel "DESCRIPCIÓN DE DATOS Rev 2.xlsx" son variables que en los modelos se utilizarán como si fuesen estados del Entorno, sin embargo, en la vida real, obedecen a condiciones de entrada externas al proceso, causadas por procesos anteriores a la torre de secado o condiciones de la planta. Estas son:

No	Variable  
2	Status_Spray_Drying  
3	Tower_PMC_Controller_Enabled  
4	Producto_A  
5	Producto_B  
6	Producto_C  
13	Bombeo_HP_TT_0355  
18	Bombeo_Slurry_Densidad  
19	Bombeo_Slurry_Humedad_HT_P401  
20	Torre_Horno_Temp_Aire  
21	Torre_Horno_Temp_Gas  


### Variables de estado del proceso

Las siguientes 16 variables, marcadas en el excel como Controlada y Manipulada, son los estados reales de la planta

No	Variable  
9	Bombeo_Low_Pump_FT_0355_Kg_h  
11	P404_High_Pump_Pressure_CV  
12	P404_High_Pump_Pressure_PV  
15	Bombeo_Aero_Boost_FT_0371_Kg_h  
17	Bombeo_Aero_Boost_PT_0371_BAR  
23	Tower_Input_Air_Fan_Speed_Feedback  
24	Flujo_de_aire_Horno  
26	Torre_Horno_Flujo_Gas  
27	Tower_Input_Temperature_PV  
29	Tower_Internal_Pressure_CV  
30	Tower_Internal_Pressure_Mean  
31	Torre_Techo_TT_0414_C  
32	Torre_PB_Flujo_Schenck  
33	Torre_PB_Humedad_MT_500  
34	F501_Ciclone_01_Speed  
35	TT5011_Dry_Cyclon_01_Temperature  


### Variables de Acción

Las siguientes ocho variables son las salidas que debe calcular la red "Red_optimizacion_secado" para optimizar económicamente la operación del proceso

No	Variable  
7	Number_of_Jets_Open  
8	Bombeo_Low_Pump_P_401  
10	P404_High_Pump_Pressure_SP  
14	Apertura_Valvula_Flujo_Aeroboost_FCV_0371  
16	Apertura_Valvula_Presion_Aeroboost  
22	Tower_Input_Air_Fan_Speed_Ref  
25	Tower_Input_Temperature_SP  
28	Tower_Internal_Pressure_SP  


Algorithm 1
Design Procedure for the proposed framework

1: Control design: Select regulatory and supervisory control layers suitable for the process dynamics, interactions, and constraints.

2: Setpoint identification: Identify setpoints with the greatest impact on operational costs and product quality using historical data and process expertise. These will become the focus of offline learning.

3: Safety override integration: Define strict safety limits for critical process variables. Implement overrides within the control layer to supersede RL-suggested setpoints that could violate these limits.

4: Offline policy learning:

• Data curation: Carefully prepare a dataset according to “Training Data and Informativity" guidelines.

• Value function learning: Choose a value function representation aligned with process complexity and data characteristics.

• Policy extraction: Apply a method such as advantage-weighted regression (AWR) to extract a policy from the learned value function, focusing on actions with high expected long-term rewards.

5: Policy deployment: Deploy the learned policy online to the system.


### Scripts Principales

- **eda.py** - Análisis exploratorio de datos
  - Genera reporte HTML con estadísticas básicas
  - Usa `ydata_profiling` para análisis rápido

- **tf_optimize.py** - Entrenamiento principal del modelo DQN
  - Implementa el agente `DQNAgent` con memoria de repetición
  - Hiperparámetros configurables
  - Guarda el modelo entrenado como `dqn_model.h5`
  - Genera gráficos de progreso del entrenamiento (`training_progress.png`)

- **test_model.py** - Validación del modelo entrenado
  - Carga el modelo y realiza pruebas unitarias
  - Compara recompensa predicha vs real
  - Manejo seguro de datos de entrada

- **cal_reward.py** - Cálculo de función de recompensa
  - Clase `RewardCalculator` con validación de datos
  - Carga pesos desde CSV (`pesos.csv`)
  - Métricas basadas en `Torre_PB_Flujo_Schenck`

- **optimize_dqn.py** - Optimización de hiperparámetros
  - Búsqueda en grid con combinaciones predefinidas
  - Guarda resultados en `optimization_results.json`
  - Parámetros optimizables: gamma, tasa de aprendizaje, tamaño de capas


### Parámetros Clave del Modelo (tf_optimize.py)

```python
HYPERPARAMS = {
    'gamma': 0.95,           # Factor de descuento
    'epsilon_init': 1.0,     # Exploración inicial
    'epsilon_min': 0.01,     # Exploración mínima
    'epsilon_decay': 0.3,    # Tasa de decaimiento de exploración
    'learning_rate': 0.001,  # Tasa de aprendizaje del optimizador
    'batch_size': 32,        # Tamaño del batch de entrenamiento
    'buffer_size': 2000,     # Capacidad máxima de la memoria
    'layer_sizes': [64, 128, 64],  # Arquitectura de la red
    'update_target_every': 100      # Pasos para actualizar red objetivo
}
