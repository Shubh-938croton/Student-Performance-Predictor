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




# test_cases = [
#     # Very Strong Student
#     [9, 95, 7, 2, 10, 90],

#     # Average Student
#     [5, 75, 6, 5, 6, 65],

#     # Weak Student
#     [2, 45, 4, 9, 2, 35],

#     # Very Weak Student
#     [1, 30, 3, 12, 1, 20],

#     # Borderline Student
#     [6, 80, 7, 4, 7, 70]
# ]

# for i, sample in enumerate(test_cases, start=1):

#     prediction = model.predict([sample])
#     probability = model.predict_proba([sample])

#     print(f"\n========== Test Case {i} ==========")
#     print("Input :", sample)
#     print("Prediction :", prediction[0])
#     print("Confidence :", round(max(probability[0]) * 100, 2), "%")
#     print("Probability :", probability)

#     import pandas as pd

# df = pd.read_csv("data/Dataset.csv")

# rows = df[df["study_hours"] == 0]

# print(rows)


import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Dataset.csv"
)

df = pd.read_csv(DATA_PATH)
print(df)

print(df["placement_status"].value_counts())

rows = df[df["study_hours"] == 0]

print(rows)

print()

print(rows["placement_status"].value_counts())


print(df["study_hours"].min())
print(df["study_hours"].max())

print(df["attendance"].min())
print(df["attendance"].max())

print(df["previous_score"].min())
print(df["previous_score"].max())


for col in [
    "sleep_hours",
    "internet_usage",
    "assignments_completed"
]:
    print(f"{col}: {df[col].min()} - {df[col].max()}")