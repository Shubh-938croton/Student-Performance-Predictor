import pandas as pd

from sklearn.preprocessing import StandardScaler  # for feature scaling 

# Load dataset
df = pd.read_csv("D:\AI- ML journey\Ml-project\Student-Performance-Predictor\Data\Dataset.csv")      # write the path correctly for readin the ambiguity.

# Display first 5 rows
print(df.head())

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

