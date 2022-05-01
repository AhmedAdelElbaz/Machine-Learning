import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv("D:/MAIN/STUDY/The Data Science Course 2021 - All Resources/1.03.+Dummies.csv")
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes':1,'No':0})
print(data.describe())
y = data['GPA']
x1 = data[["SAT", "Attendance"]]
x= sm.add_constant(x1)
results = sm.OLS(y,x).fit()
print (results.summary())
plt.scatter(data['SAT'], data['GPA'], c =data['Attendance'], cmap= 'Blues')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat = 0.0017*data['SAT'] + 0.275
fig = plt.plot(data['SAT'], yhat_no,lw = 2 , c='red', label = 'regression line1')
fig = plt.plot(data['SAT'], yhat_yes,lw = 2 , c='green', label = 'regression line2')
fig = plt.plot(data['SAT'], yhat,lw = 2 , c='blue', label = 'regression line')
plt.xlabel('SAT', fontsize= 10)
plt.ylabel('GPA', fontsize= 10)
# Create a new dataframe like the x data frame
new_data = pd.DataFrame({'const':1 , 'SAT':[1700,1760],'Attendance':[0,1]})
# perfrom predictions
predictions = results.predict(new_data)
print(predictions)
#Join the predictions to the new_data
predictionsdf = pd.DataFrame({'predictions': predictions})
joined = new_data.join(predictionsdf)
print(joined)









