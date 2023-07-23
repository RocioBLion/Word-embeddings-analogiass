import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

analogias_df = pd.read_csv('analogies_answers.csv')

#embeddings_WIKI = KeyedVectors.load_word2vec_format('embeddings/wiki.es.vec', limit=None)
embeddings_SUC = KeyedVectors.load_word2vec_format('embeddings/embeddings-l-model.vec', limit=None)
embeddings_SWOW = KeyedVectors.load_word2vec_format('embeddings/swow.embedding.was.26-04-2022.vec', limit=None)
lower_embeddings_SWOW = KeyedVectors(vector_size=embeddings_SWOW.vector_size)
#embeddings_SBWC = KeyedVectors.load_word2vec_format('embeddings/SBW-vectors-300-min5.txt', limit=None)
#embeddings_Glove = KeyedVectors.load_word2vec_format('embeddings/glove-sbwc.i25.vec', limit=None)

# Vectores y palabras del SWOW
palabras = list(embeddings_SWOW.key_to_index.keys())
vectores = list(embeddings_SWOW.vectors)

# Convertir todas las palabras a minúsculas
palabras_minusculas = [palabra.lower() for palabra in palabras]

# Crear un nuevo modelo SWOW con las palabras y vectores en minusculas
lower_embeddings_SWOW = KeyedVectors(vector_size=embeddings_SWOW.vector_size)
lower_embeddings_SWOW.add_vectors(palabras_minusculas, vectores)

lista_de_Embeddings = [embeddings_SUC, lower_embeddings_SWOW]

analogias_df['count'] = analogias_df.groupby(['cue','response'])['response'].transform('count')
analogias_df['rank'] = analogias_df.groupby('cue')['count'].rank(ascending=False, method='min')
total_respuestas = analogias_df.groupby('cue')['response'].transform('count')
analogias_df['porcentaje'] = (analogias_df['count'] / total_respuestas) * 100

# Indices de filas que tienen rango minimo (rango 1)
idx = analogias_df.groupby(['cue'])['rank'].idxmin()
# Me quedo unicamente con esas filas
analogias_df = analogias_df.loc[idx]

analogias_df.to_csv('analogies_answers.csv', index=False)

# Pasar las palabras a minusculas
analogias_df['cueEjemplo'] = analogias_df['cueEjemplo'].apply(lambda x: str(x).lower())
analogias_df['targetEjemplo'] = analogias_df['targetEjemplo'].apply(lambda x: str(x).lower())
analogias_df['cue'] = analogias_df['cue'].apply(lambda x: str(x).lower())

# Realizar los reemplazos 
analogias_df['cueEjemplo'] = analogias_df['cueEjemplo'].str.replace('lápiz labial', 'labial').replace('estados unidos', 'eeuu').replace('buenos aires', 'bsas')
analogias_df['targetEjemplo'] = analogias_df['targetEjemplo'].str.replace('lápiz labial', 'labial').replace('estados unidos', 'eeuu').replace('buenos aires', 'bsas')
analogias_df['cue'] = analogias_df['cue'].str.replace('lápiz labial', 'labial').replace('estados unidos', 'eeuu').replace('buenos aires', 'bsas')

palabras_no_encontradas = []

def calcular_predicciones(embedding, analogias_df, K):
    predicciones = []
    for index, row in analogias_df.iterrows():
        cueEjemplo = row['cueEjemplo']
        targetEjemplo = row['targetEjemplo']
        cue = row['cue']

        try:
            resultado = embedding.most_similar(positive=[targetEjemplo, cue], negative=[cueEjemplo], topn=K)
            predicciones.append([palabra for palabra, _ in resultado])
        except KeyError as e:
            print(f"Una de las palabras no está en el embedding: {e}")
            predicciones.append([])
            palabras_no_encontradas.append(str(e).split("'")[1])

    return predicciones

K = 5
todas_predicciones = [calcular_predicciones(e, analogias_df, K) for e in lista_de_Embeddings]

def calcular_topK_metricas(predicciones, analogias_df):
    analogias_df['response'] = analogias_df['response'].fillna('')
    analogias_df['response'] = analogias_df['response'].apply(lambda x: str(x))
    topK_metricas = []
    for i, prediccion in enumerate(predicciones):
        respuesta_humana = analogias_df.iloc[i]['response'].lower()
        topK_metricas.append(int(respuesta_humana in prediccion))
    return np.mean(topK_metricas)

topK_metricas_todas = [calcular_topK_metricas(prediccion, analogias_df) for prediccion in todas_predicciones]

print("Todas las metricas topK: ", topK_metricas_todas)


