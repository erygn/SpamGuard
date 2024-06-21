from flask import Flask, request, jsonify
import pickle
import unidecode

with open('model/classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

def preprocess_text(text):
    return unidecode.unidecode(text.lower())

def get_feature_importances(vectorizer, model, message):
    vect_message = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()
    class_probabilities = model.feature_log_prob_
    word_contributions = vect_message.toarray()[0] * (class_probabilities[1] - class_probabilities[0])
    contributions = sorted(zip(feature_names, word_contributions), key=lambda x: -x[1])
    return contributions

@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    vect_message = vectorizer.transform([preprocess_text(message)])
    prediction = model.predict(vect_message)[0]
    contributions = get_feature_importances(vectorizer, model, preprocess_text(message))
    return jsonify({'prediction': prediction, 'contributions': contributions})

if __name__ == '__main__':
    app.run(debug=True)
