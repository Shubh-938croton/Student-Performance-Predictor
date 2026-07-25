from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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


# create the model
model=RandomForestClassifier(random_state=42)

para_grid={
    "n_estimators":[50,100,200],
    "max_depth":[5,10,15],
    "min_samples_split":[2,5]
}

# Create the GridSearchCV object
grid_search=GridSearchCV(
    estimator=model,
    param_grid=para_grid,
    cv= 5,
    scoring="accuracy",
    n_jobs=-1     #n_jobs=-1 mean?  It tells Python: "Use all available CPU cores."

)
grid_search.fit(X,y)
print("Best Parameters:")
print(grid_search.best_params_)

print("Best Accuracy:")
print(grid_search.best_score_)

import os

print(os.cpu_count()) # 12 cores are present in my laptop 



