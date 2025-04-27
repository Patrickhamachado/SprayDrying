import json
import itertools
import numpy as np
from tf_optimize import main, set_hyperparams

def grid_search():
    # Espacio de búsqueda de hiperparámetros
    param_grid = {'gamma': [0.9, 0.95, 0.99],
                  'epsilon_decay': [0.99, 0.995, 0.999],
                  'learning_rate': [0.01, 0.001, 0.0001],
                  'batch_size': [32, 64, 128],
                  'layer_sizes': [[64, 128, 64],
                                  [128, 256, 128],
                                  [64, 64, 64]]}

    best_score = -np.inf
    best_params = {}
    results = []

    # Generar todas las combinaciones posibles
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Iniciando búsqueda en grid con {len(combinations)} combinaciones...")

    for i, params in enumerate(combinations):
        print(f"\n=== Combinación {i+1}/{len(combinations)} ===")
        print(json.dumps(params, indent=2))

        set_hyperparams(params)
        agent = main(episodes=100,
                     input_data='datos_Normal_v2_26abr_V1Filter.csv')

        if agent:
            avg_reward = np.mean(agent.episode_rewards[-20:])
            results.append({'params': params,
                            'score': avg_reward})

            if avg_reward > best_score:
                best_score = avg_reward
                best_params = params
                print("¡Nuevo mejor resultado!")

            print(f"Recompensa promedio: {avg_reward:.2f}")

    # Guardar resultados
    with open('optimization_results.json', 'w') as f:
        json.dump({'best_params': best_params,
                   'best_score': best_score,
                   'all_results': results}, f, indent=2)

    print("\n=== Optimización completada ===")
    print(f"Mejor recompensa: {best_score:.2f}")
    print("Mejores parámetros:")
    print(json.dumps(best_params, indent=2))

if __name__ == "__main__":
    grid_search()
