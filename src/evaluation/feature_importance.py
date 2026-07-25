import pandas as pd  # load data 
import matplotlib.pyplot as plt # plot the important graph

from sklearn.model_selection import train_test_split  # split the data
from sklearn.ensemble import RandomForestClassifier   # train the model 
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    r"D:\AI- ML journey\Ml-project\Student-Performance-Predictor\Data\Dataset.csv"
)

df = df.drop_duplicates()  # remove the duplicate the data.

#---------------------------------------
# Model 1 : All features 
#---------------------------------------

# features

X=df[
    [
        "study_hours",
        "attendance",
        "sleep_hours",
        "internet_usage",
        "assignments_completed",
        "previous_score"
    ]
]

 

# target 
y=df["placement_status"]

# print(df.columns)  # to print the features 

# split the data 
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#train the model 

model=RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy_all = accuracy_score(y_test, y_pred)

print("Accuracy with all features:", accuracy_all)


# find the feature importance 

# importance=model.feature_importances_

# for feature, score in zip(X.columns,importance):
#     print(f"{feature}: {score:.4f}")


# graph 

# plt.figure(figsize=(8,5))
# plt.bar(X.columns, importance)

# plt.title("Feature Importance")
# plt.xlabel("Features")
# plt.ylabel("Importance")

# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



#---------------------------------------
# Model 1 :without sleep_hours
#---------------------------------------

X_new=df[[
    "study_hours",
        "attendance",
        "internet_usage",
        "assignments_completed",
        "previous_score"
]]


X_train,X_test,y_train,y_test=train_test_split(
    X_new,
    y,
    test_size=0.2,
    random_state=42
)
model=RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy_new = accuracy_score(y_test, y_pred)

print("Accuracy without sleep_hours:", accuracy_new)

# -----------------------------
# Comparison
# -----------------------------
print("\nComparison")
print("--------------------------")
print("All Features :", accuracy_all)
print("Without sleep_hours :", accuracy_new)

if accuracy_new > accuracy_all:
    print("Removing sleep_hours improved the model.")
elif accuracy_new < accuracy_all:
    print("Removing sleep_hours reduced the model performance.")
else:
    print("No change in accuracy.")



