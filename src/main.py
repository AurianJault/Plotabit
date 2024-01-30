#!/usr/bin/python3
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn as sk

# Classification 

## KNN

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#charger les données
df=pd.read_csv('../data.csv')

# Clear datas
# alpha delta u g r i z redshift spec_OBJ_ID 
# Y : class 
x = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
y = df['class'].values


x.hist()
plt.show()

print(" Rentre un chiffre:\n\n1 - KNN\n2 - Tree\n3- RandomForestClassifier")
res = int(input())
if(res == 1):
    model = KNeighborsClassifier()
elif(res == 2):
    model = DecisionTreeClassifier(random_state=0, max_depth=20)
elif(res == 3):
    model = RandomForestClassifier(n_estimators=100 ,criterion='entropy')
else:
    raise Exception('RENTRE LE BON NOMBRE GROS CON')




Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)

Xtrain = Xtrain.values
Xtest = Xtest.values

if len(Xtrain.shape) < 2:
    Xtrain = Xtrain.reshape(-1, 1)
    
if len(Xtest.shape) < 2:
    Xtest = Xtest.reshape(-1, 1)
model.fit(Xtrain,ytrain)

ypredit = model.predict(Xtest)
# print(ypredit)
print(accuracy_score(ytest, ypredit))
