# import the Libraries 

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# read the dataset 
df=pd.read_csv(r"D:\AI- ML journey\Ml-project\Student-Performance-Predictor\Data\Dataset.csv")

# remove the duplicate 
df=df.drop_duplicates()

#select the features
X=df[
    [
        "study_hours",
        "attendance",
        "sleep_hours",
        "assignments_completed",
        "previous_score"
    ]
]

y=df["placement_status"]


# split the data into train and test 
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,test_size=0.2,
    random_state=42
)

# create the pipeline 

pipeline=Pipeline(
    steps=[                                 # steps is a list 
        ("scaler",StandardScaler()),              #Create a step called scaler using StandardScaler.       #(name,object)
        ("classifier",RandomForestClassifier(random_state=42))  # After scaling, send the data to the Random Forest.
    ]
)

# train the model 
# pipeline.fit(X_train,y_train)


# # prediction 
# predictions=pipeline.predict(X_test)


from sklearn.model_selection import cross_val_score
scores=cross_val_score(
    pipeline,
    X,
    y,
    cv=5
)


print(scores)

print(scores.mean())

#it perform worse because of scalling of data in the model random forest.
# random forest does not need data scalling.



