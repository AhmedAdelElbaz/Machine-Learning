import numpy as np
from sklearn.datasets import load_wine
data = load_wine()
X = data.data
from sklearn.cluster import KMeans
clusteringModule = KMeans(n_clusters=3,init='random')
clusteredData = clusteringModule.fit(X)
Y = data.target.reshape(-1,1)
from sklearn.metrics import adjusted_mutual_info_score
mutualInfoScore = adjusted_mutual_info_score(labels_true=clusteringModule.labels_,labels_pred=clusteringModule.labels_)
from sklearn.metrics import calinski_harabasz_score
calinski = calinski_harabasz_score(X=X,labels=clusteringModule.labels_)

from sklearn.datasets import load_boston
#----------------------------------------------------

#load boston data

BostonData = load_boston()

#X Data
X = BostonData.data

import umap
reducer = umap.UMAP(random_state=42)
reducer.fit(X)
embedding = reducer.transform(X)
assert(np.all(embedding == reducer.embedding_))
embedding.shape

import matplotlib.pyplot as plt
plt.scatter(embedding[:, 0], embedding[:, 1], c=clusteringModule.labels_, cmap='Accent', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,random_state=365,shuffle=True,train_size=0.7)
reg = LogisticRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(Y_test,Y_pred)
import matplotlib
import seaborn as sns
sns.heatmap(CM,center=True,cmap='Blues', annot=True,linewidths=0.5,linecolor='black')
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred,normalize=True)
from sklearn.metrics import recall_score
recall_score(Y_test,Y_pred,average='weighted')
from sklearn.metrics import classification_report
target_names = ['class0', 'class1', 'class2']
print(classification_report(Y_test, Y_pred, target_names=target_names))
