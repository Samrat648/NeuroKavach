from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

THRESHOLD = 0.6


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict_file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        df = pd.read_csv(file)
        df = df.select_dtypes(include=[np.number])

        if df.empty:
            return jsonify({"error": "No numeric data found"}), 400

        values = df.values.flatten()
        rms = np.sqrt(np.mean(np.square(values)))

        state = "Moment Zero" if rms > THRESHOLD else "Stable"

        return jsonify({"state": state, "rms": float(rms)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_live", methods=["POST"])
def predict_live():
    data = request.json

    if "values" not in data:
        return jsonify({"error": "No values sent"}), 400

    values = np.array(data["values"])

    rms = np.sqrt(np.mean(np.square(values)))
    state = "Moment Zero" if rms > THRESHOLD else "Stable"

    return jsonify({"state": state, "rms": float(rms)})


if __name__ == "__main__":
    app.run()