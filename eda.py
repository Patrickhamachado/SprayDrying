import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('datos_35_STD_24abr.csv')

profile = ProfileReport(df, title="Informe Exploratorio Datos Secado", explorative=True, minimal=True)

profile.to_file("informe_exploratorio_secado.html")

# https://depts.washington.edu/control/LARRY/TE/download.html
