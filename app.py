from flask import Flask, render_template, request
import joblib
import os

# -----------------------------
# Create Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Trained Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "random_forest_model.pkl"
)

model = joblib.load(MODEL_PATH)

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Read input values
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        sleep_hours = float(request.form["sleep_hours"])
        internet_usage = float(request.form["internet_usage"])
        assignments_completed = float(request.form["assignments_completed"])
        previous_score = float(request.form["previous_score"])

        # -----------------------------
        # Input Validation
        # -----------------------------
        if study_hours < 0:
            return render_template(
                "index.html",
                error="Study Hours cannot be negative."
            )

        if attendance < 0 or attendance > 100:
            return render_template(
                "index.html",
                error="Attendance must be between 0 and 100."
            )

        if sleep_hours < 0 or sleep_hours > 24:
            return render_template(
                "index.html",
                error="Sleep Hours must be between 0 and 24."
            )

        if internet_usage < 0:
            return render_template(
                "index.html",
                error="Internet Usage cannot be negative."
            )

        if assignments_completed < 0:
            return render_template(
                "index.html",
                error="Assignments Completed cannot be negative."
            )

        if previous_score < 0 or previous_score > 100:
            return render_template(
                "index.html",
                error="Previous Score must be between 0 and 100."
            )

        # -----------------------------
        # Create Feature List
        # -----------------------------
        features = [[
            study_hours,
            attendance,
            sleep_hours,
            internet_usage,
            assignments_completed,
            previous_score
        ]]

        # -----------------------------
        # Make Prediction
        # -----------------------------
        prediction = model.predict(features)

        # Prediction Probability
        probability = model.predict_proba(features)

        confidence = round(max(probability[0]) * 100, 2)

        # Convert prediction to text
        if prediction[0] == "Placed":
            result = "Placed"
        else:
            result = "Not Placed"

        # Return result to webpage
        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)