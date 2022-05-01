import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "D:/MAIN/STUDY/Data\Data/2.1 Linear Regression/satf.csv"
data = pd.read_csv(path, names=['SAT', 'GPA'])
print('data header = \n', data.head())
print("****************************")
print('Data describtion = \n' , data.describe() )
print('****************************')
data.plot(kind='scatter', x = 'SAT', y ='GPA', figsize=(5,5))
data.insert(0 , 'ones', 1)
cols = data.shape[1]
x = data.iloc[:,0:cols-1]
y = np.log(data.iloc[:,cols-1:cols])

x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
def computeCost(x,y,theta):
    z = np.power(((x * theta.T)-y),2)
    return np.sum(z) / (2*len(x))

print('computeCost = \n' , computeCost(x,y,theta))

def gradientDescent(x,y,theta,alpha,iters):
    temp= np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost= np.zeros(iters)
    for i in range(iters):
        error = (x * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(x , y , theta)

    return theta, cost

alpha = 0.05
iters = 10000

g,cost = gradientDescent(x ,y , theta, alpha, iters)

print(g)
print( 'cost = \n' , cost[0:50])
print(computeCost(x,y,g))

x = np.linspace(data.SAT.min(), data.SAT.max(),100)
f = g[0,0] + (g[0,1]*x)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x,np.exp(f),'r', label = 'Prediction')
ax.scatter(data.SAT, data.GPA, label= 'Training Data')
ax.legend(loc=2)
ax.set_xlabel('SAT')
ax.set_ylabel('GPA')
ax.set_title('Predicted Values')