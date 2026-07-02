from flask import Flask, jsonify


app = Flask(__name__)


@app.route("/")
def index():
    return jsonify(
        status="training-only",
        message=(
            "Backend inference is paused while the new VAE/classifier pipeline is "
            "being trained."
        ),
    )


@app.route("/health")
def health():
    return jsonify(status="ok", mode="training-only")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
