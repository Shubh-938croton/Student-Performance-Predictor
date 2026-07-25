import joblib
import os

# -------------------------
# Load the saved model
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "random_forest_model.pkl"
)

model = joblib.load(MODEL_PATH)

# -------------------------
# Test Data
# -------------------------

sample = [[
    9,    # Study Hours
    95,   # Attendance
    7,    # Sleep Hours
    2,    # Internet Usage
    10,   # Assignments Completed
    90    # Previous Score
]]

# -------------------------
# Prediction
# -------------------------

prediction = model.predict(sample)

probability = model.predict_proba(sample)

print("Prediction :", prediction)
print("Probability :", probability)
print("Classes :", model.classes_)

print()

print("Predicted Class :", prediction[0])

print("Confidence :", round(max(probability[0]) * 100, 2), "%")