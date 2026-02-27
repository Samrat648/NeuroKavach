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


@app.route("/predict_file", methods=["POST"])
def predict_file():
    file = request.files["file"]
    df = pd.read_csv(file)

    # Take numeric columns only
    df = df.select_dtypes(include=[np.number])

    values = df.values.flatten()

    rms = np.sqrt(np.mean(np.square(values)))

    if rms > 0.6:
        return jsonify({"result": 1, "rms": float(rms)})
    else:
        return jsonify({"result": 0, "rms": float(rms)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
