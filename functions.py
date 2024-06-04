import numpy as np

not_found_words = []  # List to store not found words

def calculate_predictions(embedding, analogies_df, fixed_K):
    predictions = []  # List to store the generated predictions
    for index, row in analogies_df.iterrows():  # Loop to iterate over the rows of the analogies_df
        cueExample = row['cueExample']
        targetExample = row['targetExample']  # Extract the key words from the dataframe row
        cue = row['cue']

        try:
            result = embedding.most_similar(positive=[cueExample, targetExample, cue], topn=fixed_K)  # Get the most similar words using Gensim's most_similar method
            predictions.append([word for word, _ in result])  # Add the similar words to the predictions list
        except KeyError as e:  # KeyError exception if any of the words are not present in the embeddings
            print(f"One of the words is not in the embedding: {e}")  # Warning message
            predictions.append([])  # Add an empty list to the predictions
            not_found_words.append(str(e).split("'")[1])  # Add the not found word to the not_found_words list
    return predictions  # Return the predictions list

def topK_metrics(predictions, analogies_df, K):
    analogies_df['response'] = analogies_df['response'].fillna('')  # Fill null values in the 'response' column
    analogies_df['response'] = analogies_df['response'].apply(lambda x: str(x))  # Ensure all values are strings
    topK_metrics = []  # List to store the calculated metrics
    for i, prediction in enumerate(predictions):  # Loop to iterate over the predictions
        human_response = analogies_df.iloc[i]['response'].lower()  # Get the human response for the current analogy and convert it to lowercase
        topK_metrics.append(int(human_response in prediction[0:K]))  # Add 1 to the list if the human response is in the predictions, and 0 otherwise
    return np.mean(topK_metrics)  # Return the mean of the topK_metrics list, representing the final top-K accuracy metric