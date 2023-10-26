import numpy as np

# Funcion para calcular las predicciones. 
palabras_no_encontradas = []
# lista vacia para las palabras no encontradas.
def calcular_predicciones(embedding, analogias_df, K_fijo):
    predicciones = []  # Lista para las predicciones.
    for index, row in analogias_df.iterrows():
        cueEjemplo = row['cueEjemplo']
        targetEjemplo = row['targetEjemplo']
        cue = row['cue']

        try:
            resultado = embedding.most_similar(positive=[targetEjemplo, cue], negative=[cueEjemplo], topn=K_fijo)
            predicciones.append([palabra for palabra, _ in resultado]) # se agregan las predicciones a la lista vacia. 
        except KeyError as e:
            print(f"Una de las palabras no está en el embedding: {e}")
            predicciones.append([]) 
            palabras_no_encontradas.append(str(e).split("'")[1]) # se agregan las palabras no encontradas a la lista vacia. 

    return predicciones

def metricas_topK(predicciones, analogias_df):
    analogias_df['response'] = analogias_df['response'].fillna('')
    analogias_df['response'] = analogias_df['response'].apply(lambda x: str(x))
    topK_metricas = []
    for i, prediccion in enumerate(predicciones):
        respuesta_humana = analogias_df.iloc[i]['response'].lower()
        topK_metricas.append(int(respuesta_humana in prediccion))
    return np.mean(topK_metricas)