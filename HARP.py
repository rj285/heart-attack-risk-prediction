import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data  = pd.read_csv("HARP.csv")

df = pd.DataFrame(data)
# print(df,df.columns) #Patient_ID,Troponin,Age,Blood Pressure,Cholesterol,Chance of Heart Attack

# plt.scatter(df['Age'],df['Chance of Heart Attack'])
# plt.xlabel("AGE")
# plt.ylabel("CHANCE OF HEART ATTACK")
# plt.savefig("HARP(dataset_plotting).png")


# Splitting data into features and target
X = df[['Troponin', 'Age', 'Blood Pressure', 'Cholesterol']]
y = df['Chance of Heart Attack']

# Creating and fitting the model
reg = LinearRegression()
reg.fit(X, y)

print("---HEART ATTACK RATE PREDICTION---")
print("EXAMPLE:- 0.03,45,110,180,30")
TL = float(input("TROPONIN:- "))
AGE = float(input("THE AGE:- "))
BP = float(input("BLOOD PRESSURE:- "))
CL = float(input("CHOLESTROL:- "))

HARP = reg.predict([[TL, AGE, BP, CL]])
# print(type(HARP)) #<class 'numpy.ndarray'>

print("Predicted Chance of Heart Attack:", HARP)