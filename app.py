from flask import Flask, jsonify, request
from classifier import getPrediction

app = Flask(__name__)


@app.route("/")
def main():
    f = open("pages/index.html")
    return f.read()


@app.route("/predict-alphabet", methods=["POST"])
def predict_data():
    image = request.files.get("alphabet")
    prediction = getPrediction(image)
    return image # jsonify({"prediction": prediction}), 200


if __name__ == "__main__":
    app.run(debug=True)
