import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Read Dataset
df = pd.read_csv(r"D:\AI- ML journey\Ml-project\Student-Performance-Predictor\Data\Dataset.csv")

# Remove duplicates
df = df.drop_duplicates()

# Features
X = df[
    [
        "study_hours",
        "attendance",
        "sleep_hours",
        "internet_usage",
        "assignments_completed",
        "previous_score"
    ]
]

# Target
y = df["placement_status"]

# Create Model 
model=RandomForestClassifier(
    n_estimators=300,
    max_depth=12
)

# Cross Validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print("Accuracy of each fold:")
print(scores)

print("Average Accuracy:")
print(scores.mean())