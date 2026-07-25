# import pandas as pd 
# import numpy as np 

# data={
#     "Hours":[2,3,np.nan,5,6],
#     "Attendance":[70,np.nan,80,90,85],
#     "Assignments":[1, 2, 3, np.nan, 5],
#     "Pass":[0,0,1,1,1]
# }

# df=pd.DataFrame(data)
# print(df)
# print(df.isnull())   #to find the missing value 

# print(df.isnull().sum())  # to fing the in each section how many values are missing.

# # df1=df.dropna()   #Remove Missing Rows

# # print(df1)

# # filling the missing Values 

# df["Hours"]=df["Hours"].fillna(df["Hours"].mean())     # at place of missing data filling the average value.

# df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())

# df["Assignments"] = df["Assignments"].fillna(df["Assignments"].mean())








# Dealing with Duplicate Data.

import pandas as pd
import numpy as np 


data = {
    "Hours": [2, 3, np.nan, 5, 6, 2, 5],
    "Attendance": [70, np.nan, 80, 90, 85, 70, 90],
    "Assignments": [1, 2, 3, np.nan, 5, 1, np.nan],
    "Pass": [0, 0, 1, 1, 1, 0, 1]
}
df=pd.DataFrame(data)
print(df)

print(df.duplicated())
print(df.duplicated().sum())

df=df.drop_duplicates()  # delete the duplicate rows 
print(df)  










