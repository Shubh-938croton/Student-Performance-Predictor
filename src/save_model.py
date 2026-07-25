
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# load data

df = pd.read_csv(
    r"D:\AI- ML journey\Ml-project\Student-Performance-Predictor\Data\Dataset.csv"
)

# remove duplicates 
df=df.drop_duplicates()

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

print(type(X))
print(type(y))

print(X.shape)
print(y.shape)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model trained successfully!")

# Save Model
joblib.dump(
    model,
    r"D:\AI- ML journey\Ml-project\Student-Performance-Predictor\src\saved_models\random_forest_model.pkl"
)

print("Model saved successfully!")




