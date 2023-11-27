import numpy as np

palabras_no_encontradas = [] # lista que va a almacenar las palabras no encontradas
def calcular_predicciones(embedding, analogias_df, K_fijo):
    predicciones = []  # lista que va a almacenar las predicciones generadas
    for index, row in analogias_df.iterrows():  # bucle que itera sobre las filas del df analogias_df
        cueEjemplo = row['cueEjemplo']
        targetEjemplo = row['targetEjemplo'] # se extraen las palabras clave de la fila del dataframe
        cue = row['cue']

        try:
            resultado = embedding.most_similar(positive=[cueEjemplo,targetEjemplo, cue], topn=K_fijo) # se obtienen las palabras más similares utilizando el método most_similar de Gensim
            predicciones.append([palabra for palabra, _ in resultado]) # se agregan las palabras similares a la lista de predicciones 
        except KeyError as e: # excepción KeyError si alguna de las palabras no está presente en los embeddings
            print(f"Una de las palabras no está en el embedding: {e}") # mensaje de advertencia
            predicciones.append([]) # se agrega una lista vacía a las predicciones
            palabras_no_encontradas.append(str(e).split("'")[1]) # se agrega la palabra no encontrada a la lista palabras_no_encontradas
    return predicciones  # devuelve la lista de predicciones.

def metricas_topK(predicciones, analogias_df,K):
    analogias_df['response'] = analogias_df['response'].fillna('') # se llenan los valores nulos en la columna 'response'
    analogias_df['response'] = analogias_df['response'].apply(lambda x: str(x)) # chequeamos de que todos los valores sean strings
    topK_metricas = [] # lista que va a almacenar las métricas calculadas 
    for i, prediccion in enumerate(predicciones): # bucle que itera sobre las predicciones 
        respuesta_humana = analogias_df.iloc[i]['response'].lower() # se obtiene la respuesta humana para la analogía actual y se la convierte a minúsculas
        topK_metricas.append(int(respuesta_humana in prediccion[0:K])) # se agrega 1 a la lista si la respuesta humana está en las predicciones, y 0 en caso contrario 
    return np.mean(topK_metricas) # devuelve el promedio de la lista topK_metricas, lo que representa la métrica final de precisión top-K 
