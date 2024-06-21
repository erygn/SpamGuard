import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import unidecode

def preprocess_text(text):
    # Convertir en minuscules et enlever les accents
    return unidecode.unidecode(text.lower())

# Charger les données depuis le fichier CSV
df = pd.read_csv('data/dataset.csv')

# Appliquer la transformation à chaque texte du dataset
df['text'] = df['text'].apply(preprocess_text)

# Prétraitement des données
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Entraînement du modèle
model = MultinomialNB()
model.fit(X, y)

# Sérialisation du modèle et du vectorizer
with open('model/classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    
with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
