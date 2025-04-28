import pandas as pd

class RewardCalculator:
    def __init__(self, pesos_path="pesos.csv"):
        self.pesos_dict = self._load_pesos(pesos_path)
        self.required_columns = self._get_required_columns()

    def _load_pesos(self, pesos_path):
        df_pesos = pd.read_csv(pesos_path)
        df_pesos['W'] = df_pesos['W'].str.replace(',', '.').astype(float)

        mapping_vars = {row['V']: f"{row['V']}" for _, row in df_pesos.iterrows()}
        df_pesos['V'] = df_pesos['V'].map(mapping_vars)

        return df_pesos.set_index('V')['W'].to_dict()

    def _get_required_columns(self):
        return list(self.pesos_dict.keys())

    def validate_data(self, df_data):
        missing = [col for col in self.required_columns if col not in df_data.columns]
        if missing:
            raise ValueError(f"Columnas requeridas faltantes: {missing}")
        return True

    def calculate_reward(self, row):
        self.validate_data(row.to_frame().T)
        base_var = 'Torre_PB_Flujo_Schenck'

        try:
            base_weight = self.pesos_dict[base_var]
            base = row[base_var] * base_weight
            componentes_negativos = sum(row[var] * peso
                                    for var, peso in self.pesos_dict.items()
                                    if var != base_var)
            return base - componentes_negativos
        except KeyError as e:
            print(f"Error en cálculo de recompensa: {str(e)}")
            print("Columnas disponibles:", row.index.tolist())
            return 0  # Valor por defecto si hay error

    def add_reward_column(self, df_data):
        self.validate_data(df_data)
        df_data['reward'] = df_data.apply(self.calculate_reward, axis=1)
        return df_data

if __name__ == "__main__":
    calculator = RewardCalculator("data/pesos.csv")
    df_data = pd.read_csv("data/sample_data.csv")

    try:
        df_with_reward = calculator.add_reward_column(df_data)
        df_with_reward.to_csv('sample_data_con_reward.csv', index=False)
        print("Recompensas calculadas exitosamente")
    except Exception as e:
        print(f"Error: {str(e)}")
