import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('respuestas_finales2.csv')

# Calcular el total de respuestas por estimulo
total_respuestas = df.groupby('estimulo')['count'].sum()

# Calcular el porcentaje de respuestas para cada fila
df['porcentaje'] = df.apply(lambda row: row['count'] / total_respuestas[row['estimulo']] * 100, axis=1)

# Guardar los cambios en un nuevo archivo CSV
df.to_csv('respuestas_finales2.csv', index=False)

df