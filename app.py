from flask import Flask, request, render_template
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        transformed_msg = vectorizer.transform([message])
        prediction = model.predict(transformed_msg)[0]
        result = "Spam ❌" if prediction else "Not Spam ✅"
        return render_template('index.html', prediction=result, user_input=message)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
