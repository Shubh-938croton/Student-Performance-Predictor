import pandas as pd

from sklearn.preprocessing import StandardScaler  # for feature scaling 

# Load dataset
df = pd.read_csv("D:\AI- ML journey\Ml-project\Student-Performance-Predictor\Data\Dataset.csv")      # write the path correctly for readin the ambiguity.

# Display first 5 rows
print(df.head(10))

# Display dataset information
print(df.info())

# Display statistical summary
print(df.describe())
print(df.columns)
print(df.shape)


print(df.isnull().sum())

print(df.duplicated().sum())

df=df.drop_duplicates() # removes the duplicates 

# first feature selection 
# X is input data 

# regression problem
X=df[[
    "study_hours",
    "attendance",
    "sleep_hours",
    "internet_usage",
    "assignments_completed",
    "previous_score"
]
]
# y is output data to be predicted by the model 
y=df["exam_score"]

# second feature selection 
# X is for input 
# classification problem 
X = df[
    [
        "study_hours",
        "attendance",
        "sleep_hours",
        "internet_usage",
        "assignments_completed",
        "previous_score",
        "exam_score"
    ]
]
# y is for predicting the placement status 


y=df["placement_status"]

scalar=StandardScaler()  # create the StandardScalar object.

X_scaled=scalar.fit_transform(X) # it perform two operations i- fit() studies your dataset.ii transform() every value is converted using the standardization formula.


X_scaled=pd.DataFrame(  # this convert the data back into a DataFrame:
    X_scaled,
    columns=X.columns  #"Use the same column names that the original DataFrame had."
)
print(X_scaled.head())  # shows the first five rows after scaling 

print(X.head()) # before scaling 
print(X_scaled.head()) # after scaling 

print(df.groupby("placement_status")["exam_score"].describe())





# trian_test_split 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,   # 20% for testing 
    random_state=42  # for getting same train/test split in case of running on same data.

    
)
print("Training Features:", X_train.shape)
print("Testing Features:",X_test.shape)

print("Training target:",y_train.shape)
print("Testing Target:",y_test.shape)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

predictions=model.predict(X_test)

print("Actual Values:")
print(y_test.values[:10])

print("Predicted Values:")
print(predictions[:10])


# evaluation of the data using Random Forest model. 

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)

print("Accuracy:",accuracy)

# Confusion Matrix

from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test,predictions)
print("Confusion Matrix:",cm)


from sklearn.metrics import(
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


precision=precision_score(y_test,predictions,pos_label="Placed")  # pos_lebel means targetting the string "Placed".
recall=recall_score(y_test,predictions,pos_label="Placed")
f1=f1_score(y_test,predictions,pos_label="Placed")

print("Precision:",predictions)
print("Recall:",recall)
print("F1 Score:",f1)

print("\nClassification Report")
print(classification_report(y_test, predictions))

