import pandas as pd

class RewardCalculator:
    def __init__(self, pesos_path="pesos.csv"):
        self.pesos_dict = self._load_pesos(pesos_path)

    def _load_pesos(self, pesos_path):
        df_pesos = pd.read_csv(pesos_path)
        df_pesos['W'] = df_pesos['W'].str.replace(',', '.').astype(float)

        mapping_vars = {row['V']: f"{row['V']}" for _, row in df_pesos.iterrows()}
        df_pesos['V'] = df_pesos['V'].map(mapping_vars)

        return df_pesos.set_index('V')['W'].to_dict()

    def calculate_reward(self, row):
        base_var = 'Torre_PB_Flujo_Schenck'
        base_weight = self.pesos_dict[base_var]

        base = row[base_var] * base_weight
        componentes_negativos = sum(row[var] * peso
                                    for var, peso in self.pesos_dict.items()
                                    if var != base_var)

        return base + componentes_negativos

    def add_reward_column(self, df_data):
        df_data['reward'] = df_data.apply(self.calculate_reward, axis=1)
        return df_data

if __name__ == "__main__":
    # Crear instancia del calculador
    calculator = RewardCalculator("pesos.csv")

    # Cargar datos y calcular recompensa
    df_data = pd.read_csv("datos_Normal_v2_26abr_V1Filter.csv")
    df_with_reward = calculator.add_reward_column(df_data)

    # Guardar resultados
    df_with_reward.to_csv('sample_data_con_reward.csv', index=False)
