from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# MODE 1: FILE UPLOAD
@app.route("/predict_file", methods=["POST"])
def predict_file():
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        if df.shape[1] < 3:
            return jsonify({"error": "CSV must have at least 3 columns"}), 400

        features = [df.mean().values[:3]]
        prediction = model.predict(features)[0]

        return jsonify({"result": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# MODE 2: LIVE ESP DATA
@app.route("/predict_live", methods=["POST"])
def predict_live():
    try:
        data = request.json["values"]

        if len(data) < 3:
            return jsonify({"error": "Need at least 3 values"}), 400

        prediction = model.predict([data[:3]])[0]

        return jsonify({"result": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
if __name__ == "__main__":
    app.run()
