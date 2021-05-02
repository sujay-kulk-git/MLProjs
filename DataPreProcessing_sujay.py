#Data Preprocessing
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 3].values

#missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])

#divide Country to 3 columns to avoid false-preference issue
onehotencoder = OneHotEncoder(categorical_features = [0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y= labelencoder_X.fit_transform(y)

#Training and Test set data split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling- to avoid large numbers dominating Eucleadian distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
#fir the model to train set and use same to transform test set
X_test=sc_X.transform(X_test)

