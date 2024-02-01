#!/usr/bin/python3
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn as sk

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Open dataset with panda
def read_dataset(filename):
    df = pd.read_csv(filename)
    return df

# Drop useless columns and return x and y
def get_xy_from_dataset(filename):
    df = read_dataset(filename)
    x = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
    y = df['class'].values
    return x, y 
    
x, y = get_xy_from_dataset("data.csv")


x.hist()
#plt.show()

print("""Choose a model:
(1) - KNN
(2) - Tree
(3) - RandomForestClassifier
(4) - SGD
(5) - Linear SVC""")
res = int(input())

if (res == 1):
    model = KNeighborsClassifier()
elif (res == 2):
    model = DecisionTreeClassifier(random_state=0, max_depth=20)
elif (res == 3):
    model = RandomForestClassifier(n_estimators=100 ,criterion='entropy')
elif (res == 4):
    model = SGDClassifier(max_iter=1000, tol=0.01)
elif (res == 5):
    model = svm.SVC(kernel='linear', C = 1.0)
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
