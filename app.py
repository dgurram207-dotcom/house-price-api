from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        area = float(request.form["area"])
        bedrooms = float(request.form["bedrooms"])
        age = float(request.form["age"])
        distance = float(request.form["distance"])

        features = np.array([[area, bedrooms, age, distance]])
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Price: ₹ {round(prediction,2)} Lakhs"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)