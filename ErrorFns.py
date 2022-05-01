import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("D:/MAIN/Samples/counts.csv")

'''
Z = []
for i in range(1,2000):
    for j in range(1,500):
        z = np.sqrt(np.power(data.iloc[j:,i],2) + np.power(data.iloc[j:,i+1],2))
        i +=2
        Z.append(z)
Z = np.array([Z])
Z = pd.DataFrame(Z)
Z['']= data.iloc[:,0]
Z.shape
data.describe()
'''
cols = data.shape[1]
y = data.iloc[:, cols-1]
x = data.iloc[: , 1:cols-1]
x , y = np.log((x+0.0000001)) , np.log((y+0.0000001))
x = np.array([[x]]).reshape(x.shape[1],-1)
sns.distplot(y)
y = y <= 1.0
for i in range(0,100):
    sns.scatterplot(x[i], y, x="X", y = "y")
y = y <= y.quantile(0.25)
y = np.array([[y]]).reshape(-1,1)
#data splitting
x_train ,x_test , y_train , y_test = train_test_split(x,y, test_size=0.3, random_state= 99)
reg = LinearRegression()
reg1 = LogisticRegression()
reg1.fit(x_train,y_train)
n = [range(0,len(y))]
residuals = reg1.predict(x_test)- y_test
error = 0
for i in range(0,len(y_test)):
    if reg1.predict(x_test)[i] == y_test:
        error +=0
    else:
        error+=1
test_error = (1/y_test)*error
data.isnull().sum()
len(y_test)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_train)
y_pred[0:10,]
y_test[0:10,]
J = (1/len(y_test))* np.sum(np.power((y_pred-y_test),2))+ (0.0001 * np.sum(np.power((reg.coef_),2)))
print("******************************************","\n","R squared =","\t",reg.score(x,y),"\n","******************************************")
reg.score(x_train,y_train)
reg.coef_
reg.score(x,y)
plt.scatter(y_pred, y)
def plot_line(x,y,y_pred, xlabel="X-axis", ylabel="Y-Axis"):
    plt.scatter(x , y , c ='b', s = 3)
    plt.plot(x , y_pred, c = 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
plot_line(x,y,y_pred)

def variance_of_residuals(y,y_pred):
    print("******************************************","\n","Variance of Residuals =")
    return np.sum(np.power((y-y_pred)-np.average(y - y_pred),2))/(len(y)-1)
variance_of_residuals(y,y_pred)

def Standard_deviation_of_residuals(y,y_pred):
    print("******************************************", "\n", "Standard Deviation of Residuals =")
    return np.sqrt(np.sum(np.power((y - y_pred) - np.average(y - y_pred), 2)) / (len(y) - 1))
Standard_deviation_of_residuals(y,y_pred)

def confidence_interval(x,y,y_pred,Z):
    positive , negative  = np.average(y - y_pred) + Z*(Standard_deviation_of_residuals(y,y_pred)/np.sqrt(len(y))) , np.average(y - y_pred) - Z * (Standard_deviation_of_residuals(y, y_pred) / np.sqrt(len(y)))
    print("******************************************", "\n", "Confidence Interval=","\n","[",positive,",",negative,"]")
    if x.shape[1]== 1:
        plt.plot(x,y_pred, c ="r")
        plt.plot(x, y_pred + positive, c="g", linestyle = "solid")
        plt.plot(x, y_pred + negative, c="black", linestyle = "solid")
        plt.scatter(x, y, c='b', s =10)
        plt.xlabel("Score")
        plt.ylabel("GPA")
        plt.show()
    else:
        print("No plot")
confidence_interval(x,y,y_pred,1.96)

def standard_error_of_regression(y,y_pred):
    print("******************************************", "\n", "Standard Error of Regression =")
    #return np.sqrt(1/ (len(y)-2)*np.sum(np.power((y-y_pred),2)) / np.sum(np.power(y-(np.average(y)),2)))
    return np.sqrt(1 / (len(y) - 2) * np.sum(np.power((y - y_pred), 2)) )
standard_error_of_regression(y,y_pred)

def RSS(y,y_pred):
    print("******************************************", "\n", "RSS =")
    return np.sum(np.power(y-y_pred,2))
RSS(y,y_pred)

def ESS(y,y_pred):
    print("******************************************", "\n", "ESS =")
    return np.sum(np.power(y_pred-np.average(y),2))
ESS(y,y_pred)

def MSE(y,y_pred):
    print("******************************************", "\n", "MSE =")
    return (np.sum(np.power(y-y_pred,2)))/len(y)
MSE(y,y_pred)

def RMSE(y,y_pred):
    print("******************************************", "\n", "RMSE =")
    return np.sqrt((np.sum(np.power(y-y_pred,2)))/len(y))
RMSE(y,y_pred)

def MAE(y,y_pred):
    print("******************************************", "\n", "MAE =")
    return (np.sum(np.abs(y-y_pred)))/len(y)
MAE(y,y_pred)

def R_squared(y,y_pred):
    print("******************************************", "\n", "R_Squared =")
    return 1 -np.sum(np.power(y-y_pred,2))  /(np.sum(np.power(y-y_pred,2)) + np.sum(np.power(y_pred-np.average(y),2)))
R_squared(y,y_pred)

def Adjusted_R(x,y,y_pred):
    print("******************************************", "\n", "Adjusted R Squared =")
    return 1 - (((1-1 -np.sum(np.power(y-y_pred,2))  /(np.sum(np.power(y-y_pred,2)) + np.sum(np.power(y_pred-np.average(y),2))))*(len(y)-1))/(len(y)-x.shape[1]-1))
Adjusted_R(x,y,y_pred)

