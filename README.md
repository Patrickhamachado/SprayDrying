# Proyecto de Optimización de Secado con DQN

Sistema de aprendizaje por refuerzo profundo para optimización del proceso de secado en la industria minera.

## Estructura del Proyecto

### Scripts Principales

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

- **eda.py** - Análisis exploratorio de datos
  - Genera reporte HTML con estadísticas básicas
  - Usa `ydata_profiling` para análisis rápido

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
