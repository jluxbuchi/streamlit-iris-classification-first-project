# import Libraries
 
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Read Dataset
df = pd.read_csv('iris 3.csv')
 
# Display dataset
print(df.head())
 
# Display Column Names
print(df.columns)
 
# Display Shape
print(df.shape)
 
# Display unique values from column variety
print(df['variety'].unique())               # ['Setosa', 'Versicolor', 'Virginica']


from sklearn.preprocessing import StandardScaler

# Step - 4 : Data Preprocessing

 

# Replace ['Setosa' : 0, 'Versicolor' : 1 , 'Virginica' : 2 ]

df['target'] = df['variety'].map(   { 'Setosa' : 0, 'Versicolor' : 1 , 'Virginica' : 2 }  )

 

# Decide I/P and O/P

 

X = df.drop(['target', 'variety'] , axis = 1 )

Y = df['target']


 # Step - 4 : Data Preprocessing

 

# Replace ['Setosa' : 0, 'Versicolor' : 1 , 'Virginica' : 2 ]

df['target'] = df['variety'].map(   { 'Setosa' : 0, 'Versicolor' : 1 , 'Virginica' : 2 }  )

# Decide I/P and O/P

 

X = df.drop(['target', 'variety'] , axis = 1 )    # Input / Independant Features

Y = df['target']                                  # Output / Dependant Features

# APPLY STANDARD SCALER

 

SS = StandardScaler()

X_ss = SS.fit_transform(X)

# STEP -5  : SPLIT DATASET

from sklearn.model_selection import train_test_split

 

X_train, X_test, Y_train, Y_test = train_test_split(X_ss, Y, test_size= 0.20,random_state= 123)

# Step 6 : Apply Classification Algorithm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

RFC = RandomForestClassifier()

RFC.fit(X_train, Y_train)

 

# Evaluation

print("\n\n Accuracy Score : ", accuracy_score(Y_test, RFC.predict(X_test)))

# Save Model

 

import pickle

 

pickle.dump( RFC  , open("model.pkl" , "wb"))

pickle.dump(  SS  ,open("scaler.pkl", "wb") )

