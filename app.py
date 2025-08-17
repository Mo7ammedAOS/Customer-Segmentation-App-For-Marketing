from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("./model/models.pkl", "rb") as f:
    model = pickle.load(f)

with open("./model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


with open("./model/features_names.pkl", "rb") as f:
    features_names = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        features = [float(value) for value in request.form.values()]
        features_scaled = scaler.transform([features])
        cluster = model.predict(features_scaled)[0]
        return render_template("result.html", cluster=cluster)
    return render_template("home.html", features_names=features_names)

if __name__ == "__main__":
    app.run(debug=True)
