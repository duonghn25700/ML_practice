import pandas as pd 
import numpy as np 
from sklearn import linear_model
import math

df = pd.read_csv("homeprices.csv")
# print(df)

median_bedrooms = math.floor(df.bedrooms.median())

df.bedrooms = df.bedrooms.fillna(median_bedrooms)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']], df.price)

pred = reg.predict([[3000, 3, 40]])
print(pred)