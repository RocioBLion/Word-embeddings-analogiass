import unittest
from gensim.models import KeyedVectors
import pandas as pd
from mis_funciones import metricas_topK, calcular_predicciones

class TestMetricasTopK(unittest.TestCase):
    def setUp(self):
        self.embedding_ejemplo = KeyedVectors.load_word2vec_format('embeddings/wiki.es.vec', limit=None)
        self.analogias_df_ejemplo = pd.read_csv('analogies_answers.csv')
        self.todas_predicciones = [calcular_predicciones(e, self.analogias_df_ejemplo, K=5) for e in [self.embedding_ejemplo]]
        
    def test_calcular_predicciones(self):
        # calcular_predicciones con los valores de prueba
        predicciones = calcular_predicciones(self.embedding_ejemplo, self.analogias_df_ejemplo, K=5) #toma las 3 entradas necesarias

        # Verificar si el resultado es el esperado
        self.assertEqual(len(predicciones), len(self.analogias_df_ejemplo)) #Verificar que la longitud de la lista de predicciones generada sea igual al número de filas en analogias_df, lo que significa que se generaron predicciones para cada fila de datos de analogías.
        
    def test_metricas_topK(self):
        # definir las métricas top-K esperadas para las predicciones 
        valores_esperados = [0.17567567567567569]  # metrica obtenida para WIKI
        # toma un conjunto de predicciones como entrada

        # Calcula las métricas top-K tomando las predicciones y la resp humana
        topK_metrica = metricas_topK(self.todas_predicciones[0], self.analogias_df_ejemplo)

        self.assertAlmostEqual(topK_metrica, valores_esperados[0], delta=1e-6)  # Compara los valores esperados y el obtenido

if __name__ == '__main__':
    unittest.main()

