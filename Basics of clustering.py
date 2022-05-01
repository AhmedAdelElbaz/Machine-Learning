import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

# Open the file as pandas dataframe
filepath = "D:/MAIN/STUDY/The Data Science Course 2021 - All Resources/"
data = pd.read_csv(filepath + "3.01.+Country+clusters.csv")
print(data)
#Sctter plot the numeric values
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
#separate the numeric values for clustering
x = data.iloc[:,1:3]
print(x)
#Set the number of wanted clusters
Kmeans = KMeans(3)
Kmeans.fit(x)
#the identified clusters
identified_clusters = Kmeans.fit_predict(x)
print(identified_clusters)
#add the clusters column
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters
print(data_with_clusters)
#Sctter plot the numeric values
plt.scatter(data['Longitude'],data['Latitude'], c= data_with_clusters['Clusters'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
#get the within clusters sum of squares
wcss = []
for i in range(1,7):
    Kmeans =KMeans(i)
    Kmeans.fit(x)
    wcss_iter = Kmeans.inertia_
    wcss.append(wcss_iter)
print(wcss)
#create the elbow plot
number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('Within-cluster Sum of sqaure')
plt.show()