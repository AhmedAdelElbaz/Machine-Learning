from sklearn.impute import SimpleImputer
import numpy as np

imputerModel = SimpleImputer(missing_values=0,strategy='constant', fill_value=8)
X = np.array([[0,1],[0,2],[3,4]])
imputed = imputerModel.fit(X)
X = imputed.transform(X)
