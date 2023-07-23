import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

analogias_df = pd.read_csv('analogies_answers.csv')
embeddings_WIKI = KeyedVectors.load_word2vec_format('embeddings/wiki.es.vec', limit=None)
embeddings_SUC = KeyedVectors.load_word2vec_format('embeddings/embeddings-l-model.vec', limit=None)
embeddings_SBWC = KeyedVectors.load_word2vec_format('embeddings/fasttext-sbwc.3.6.e20.vec', limit=None)
embeddings_Glove = KeyedVectors.load_word2vec_format('embeddings/glove-sbwc.i25.vec', limit=None)

analogias_df['count'] = analogias_df.groupby(['cue','response'])['response'].transform('count')
analogias_df['rank'] = analogias_df.groupby('cue')['count'].rank(ascending=False, method='min')
total_respuestas = analogias_df.groupby('cue')['response'].transform('count')
analogias_df['porcentaje'] = (analogias_df['count'] / total_respuestas) * 100

#índices de filas que tienen rango mínimo (rango 1)
idx = analogias_df.groupby(['cue'])['rank'].idxmin()
# me quedo únicamente con esas filas
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

# Obtener las palabras de las columnas preparadas
column1_palabras = set(analogias_df['cueEjemplo'].unique())
column2_palabras = set(analogias_df['targetEjemplo'].unique())
column3_palabras = set(analogias_df['cue'].unique())
palabras_combinadas = column1_palabras.union(column2_palabras, column3_palabras)

# Lista para las palabras que no están en el embedding
palabras_no_en_embedding = []

# Verificar si cada palabra está en el embedding
for palabra in palabras_combinadas:
    if palabra not in embeddings_SBWC.key_to_index:
        palabras_no_en_embedding.append(palabra)


# Palabras que no están en el embedding
print(palabras_no_en_embedding)

# TODAS LAS PREDICCIONES

# Predicciones del modelo para cada analogía

#predicciones_SUC = []
#for index, row in analogias_df.iterrows():
    #cueEjemplo = row['cueEjemplo']
    #targetEjemplo = row['targetEjemplo']
    #cue = row['cue']

    #try:
        #resultado_SUC = embeddings_SUC.most_similar(positive=[targetEjemplo, cue], negative=[cueEjemplo], topn=5)
        #predicciones_SUC.append([palabra for palabra, _ in resultado_SUC])
    #except KeyError as e:
        #print(f"Una de las palabras no está en el embedding: {e}")
        #predicciones_SUC.append([])  # Se agrega a una lista vacia


#predicciones_Glove = []
#for index, row in analogias_df.iterrows():
    #cueEjemplo = row['cueEjemplo']
    #targetEjemplo = row['targetEjemplo']
    #cue = row['cue']

    #try:
        #resultado_Glove = embeddings_Glove.most_similar(positive=[targetEjemplo, cue], negative=[cueEjemplo], topn=5)
        #predicciones_Glove.append([palabra for palabra, _ in resultado_Glove])
    #except KeyError as e:
        #print(f"Una de las palabras no está en el embedding: {e}")
        #predicciones_Glove.append([])  # Se agrega a una lista vacia

predicciones_WIKI = []
for index, row in analogias_df.iterrows():
    cueEjemplo = row['cueEjemplo']
    targetEjemplo = row['targetEjemplo']
    cue = row['cue']

    try:
        resultado_WIKI = embeddings_WIKI.most_similar(positive=[targetEjemplo, cue], negative=[cueEjemplo], topn=5)
        predicciones_WIKI.append([palabra for palabra, _ in resultado_WIKI])
    except KeyError as e:
        print(f"Una de las palabras no está en el embedding: {e}")
        predicciones_WIKI.append([])  # Se agrega a una lista vacia


#predicciones_SBWC = []
#for index, row in analogias_df.iterrows():
    #cueEjemplo = row['cueEjemplo']
    #targetEjemplo = row['targetEjemplo']
    #cue = row['cue']

    #try:
        #resultado_SBWC = embeddings_SBWC.most_similar(positive=[targetEjemplo, cue], negative=[cueEjemplo], topn=5)
        #predicciones_SBWC.append([palabra for palabra, _ in resultado_SBWC])
    #except KeyError as e:
        #print(f"Una de las palabras no está en el embedding: {e}")
        #predicciones_SBWC.append([])  # Se agrega a una lista vacia
        
        
# TODAS LAS PREDICCIONES


# Top 5 respuestas humanas para cada analogía
respuestas_humanas_top5_df = analogias_df.groupby('cue')['response'].apply(lambda x: x.head(5).tolist())
#print(respuestas_humanas_top5_df)

# Calcular la similitud del coseno
def calculo_similitud(embedding_modelo, palabra1, palabra2):
    if palabra1 in embedding_modelo and palabra2 in embedding_modelo:
        vector1 = embedding_modelo[palabra1].reshape(1, -1)
        vector2 = embedding_modelo[palabra2].reshape(1, -1)
        return cosine_similarity(vector1, vector2)[0][0]
    else:
        return None

similitud_dict = {}
predicciones_dict = {}
predicciones_humanas_dict = {}

# Calculo de las similitudes para cada analogía
for index, row in analogias_df.iterrows():
    analogia = row['cue']
    predicciones = predicciones_WIKI[index] if index < len(predicciones_WIKI) else []
    respuestas_humanas = respuestas_humanas_top5_df[analogia] if analogia in respuestas_humanas_top5_df else []
    
    predicciones_dict[analogia] = predicciones
    predicciones_humanas_dict[analogia] = respuestas_humanas
    
    similitudes = []
    for pred, resp in zip(predicciones, respuestas_humanas):
        similarity = calculo_similitud(embeddings_Glove, pred, resp)
        if similarity is not None:
            similitudes.append(similarity)

    if similitudes:
        similitud_dict[analogia] = np.mean(similitudes)

# Resultados
for analogia in analogias_df['cue'].unique():
    if analogia in similitud_dict:
        print(f"Analogía: {analogia}, Predicciones del modelo: {predicciones_dict[analogia]}, Predicciones humanas: {predicciones_humanas_dict[analogia]}, Similitud Promedio: {similitud_dict[analogia]}")
