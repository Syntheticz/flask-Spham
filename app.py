import numpy as np
from flask import Flask, abort, jsonify, request
import joblib


# Load the spam detection model
loaded_model = joblib.load('spam_model.joblib')
vectorizer = joblib.load('vectorizer.joblib') 
threshold = 0.6479933191798792


app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)
    text = data['text']
    test_text_transformed = vectorizer.transform([text])
    probabilities = loaded_model.predict_proba(test_text_transformed)
    spam_probability = probabilities[0, 1]
    prediction = loaded_model.predict(test_text_transformed)
    threshold = 0.6479933191798792
  
    if spam_probability >= threshold:
        return jsonify({ "category" : "spam"})
    else:
         return jsonify({ "category" : "ham"})

if __name__ == '__main__':
    app.run(port=8000, debug=True,host="localhost")


