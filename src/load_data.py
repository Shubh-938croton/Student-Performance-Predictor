import pandas as pd 

df=pd.read_csv("Data/student_dataset_10000_rows.csv")
print(df.head)
print(df.describe)
print(df.info())
