import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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


param_dist = {
    "n_estimators": [100,200,300,400,500],
    "max_depth": [5,10,15,20,25],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
}


random_search=RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)
random_search.fit(X,y)
print("Best Parameters:")
print(random_search.best_params_)

print("Best Accuracy:")
print(random_search.best_score_)