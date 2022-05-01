import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

data = pd.read_csv("D:/MAIN/STUDY/The Data Science Course 2021 - All Resources/1.02.+Multiple+linear+regression.csv")

x= data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

#standardization ising the StandardScaler() from Sklearn
from sklearn.preprocessing import StandardScaler

#Create Empty standard scaler object
scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)

# to apply it for a new created data:
new_data = pd.read_csv("D:/MAIN/STUDY/The Data Science Course 2021 - All Resources/1.02.+Multiple+linear+regression.csv")
scaler1 = StandardScaler()
scaler1.fit(new_data)
scaler1 = scaler1.transform(new_data)

#fiiting the regression
reg = LinearRegression()
reg.fit(x_scaler, y)
coeff = reg.coef_
intercept = reg.intercept_
p_values = f_regression(x_scaler,y)[1]
reg_summery = pd.DataFrame([['Bias'], ['SAT'], ['Rand 1,2,3']], columns= ['Features'])
reg_summery['weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
print (reg_summery)
 #Creating new data frame

new_data1 = pd.DataFrame(data = [[1700,2],[1800,1]], columns = ['SAT', 'Rand 1,2,3'])
print(new_data1)
new_data_scaled = scaler.transform(new_data1)
predictions = reg.predict(new_data_scaled)
new_data1['preicted GPA'] = predictions
print(new_data1)

#removed Rand 1,2,3
reg_simple = LinearRegression()
x_simple_matrix = x_scaler[:,0].reshape(-1,1)
reg_simple.fit(x_simple_matrix,y)
predictions = reg_simple.predict(new_data_scaled[:,0].reshape(-1,1))
new_data1['predicted GPA #']= predictions
print(new_data1)

