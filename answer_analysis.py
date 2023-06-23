import pandas as pd
from gensim.models import KeyedVectors

analogias_df = pd.read_csv('analogies_answers.csv')
#embeddings_WIKI = KeyedVectors.load_word2vec_format('embeddings/wiki.es.vec', limit=None)
#embeddings_SUC = KeyedVectors.load_word2vec_format('embeddings/embeddings-l-model.vec', limit=None)
#embeddings_SBWC = KeyedVectors.load_word2vec_format('embeddings/fasttext-sbwc.3.6.e20.vec', limit=None)
embeddings_GloVe = KeyedVectors.load_word2vec_format('embeddings/glove-sbwc.i25.vec', limit=None)


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
    if palabra not in embeddings_GloVe.key_to_index:
        palabras_no_en_embedding.append(palabra)

# Palabras que no están en el embedding
print(palabras_no_en_embedding)
