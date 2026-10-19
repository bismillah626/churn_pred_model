import os

from flask import Flask, jsonify, render_template, request

from src.dashboard_service import get_dashboard_data
from src.pipeline.testing_pipeline import PredictionPipeline


application = Flask(__name__)

app = application


def _prediction_inputs(source):
    """Validate and normalize the fields shared by the web and JSON forms."""

    submitted = {
        "age": source.get("age"),
        "gender": source.get("gender") or source.get("Gender"),
        "tenure": source.get("tenure"),
        "CreditScore": source.get("CreditScore"),
        "IsActiveMember": source.get("IsActiveMember"),
        "Geography": source.get("Geography"),
        "NumOfProducts": source.get("NumOfProducts"),
    }
    missing = [field for field, value in submitted.items() if value is None or value == ""]
    if missing:
        raise ValueError(f"Please provide: {', '.join(missing)}")

    try:
        age = int(source.get("age"))
        tenure = int(source.get("tenure"))
        credit_score = int(source.get("CreditScore"))
        active_member = int(source.get("IsActiveMember"))
    except (TypeError, ValueError) as error:
        raise ValueError("Age, tenure, credit score, and active member must be numbers") from error

    if not 18 <= age <= 100:
        raise ValueError("Age must be between 18 and 100")
    if not 0 <= tenure <= 10:
        raise ValueError("Tenure must be between 0 and 10")
    if not 300 <= credit_score <= 850:
        raise ValueError("Credit score must be between 300 and 850")
    if active_member not in (0, 1):
        raise ValueError("Active member must be either 0 or 1")

    return {
        "Age": age,
        "Gender": submitted["gender"],
        "Tenure": tenure,
        "CreditScore": credit_score,
        "IsActiveMember": active_member,
        "Geography": source.get("Geography"),
        "NumOfProducts": source.get("NumOfProducts"),
    }


def _run_prediction(source):
    values = _prediction_inputs(source)
    prediction_obj = PredictionPipeline(**values)
    prediction, shap_img = prediction_obj.predict_and_explain()
    return int(prediction[0]), shap_img


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/dashboard")
def dashboard_data():
    """Return the cached analytics payload used by the interactive dashboard."""

    try:
        return jsonify(get_dashboard_data())
    except Exception as error:
        app.logger.exception("Unable to build dashboard analytics")
        return jsonify({"error": str(error)}), 500


@app.post("/api/predict")
def predict_api():
    try:
        prediction, shap_img = _run_prediction(request.get_json(silent=True) or {})
        return jsonify({"prediction": prediction, "shap_plot": shap_img})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The model could not score this customer. Check the server log."}), 500


@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")

    try:
        prediction, shap_img = _run_prediction(request.form)
        return render_template("home.html", result=prediction, shap_plot=shap_img)
    except ValueError as error:
        return render_template("home.html", error=str(error)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template(
            "home.html",
            error="The model could not score this customer. Check the server log.",
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=os.getenv("FLASK_DEBUG", "0") == "1")
