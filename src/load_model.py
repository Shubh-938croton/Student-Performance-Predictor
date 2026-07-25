import joblib
from pathlib import Path

# find the model 

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR /"saved_models"/"random_forest_model.pkl"

model=joblib.load(MODEL_PATH)
print("Model loaded successfully!")
print(model)

# new student data to predict 

new_student=[
    [
        8,
        90,
        7,
        2,
        5,
        82
    ]
]

prediction=model.predict(new_student)

print(prediction)


# study_hours = float(input("Enter Study Hours: "))
# attendance = float(input("Enter Attendance: "))
# sleep_hours = float(input("Enter Sleep Hours: "))
# internet_usage = float(input("Enter Internet Usage: "))
# assignments_completed = float(input("Enter Assignments Completed: "))
# previous_score = float(input("Enter Previous Score: "))

# new_student = [[
#     study_hours,
#     attendance,
#     sleep_hours,
#     internet_usage,
#     assignments_completed,
#     previous_score
# ]]

# prediction = model.predict(new_student)

# print("Prediction:", prediction[0])

probability=model.predict_proba(new_student)

print(probability)