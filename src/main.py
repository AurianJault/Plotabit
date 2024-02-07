#!/usr/bin/python3
import os
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
# from sklearn.externals.joblib import parallel_backend

# main
def main():
    # User input
    opt = prompt_display()
    model = model_switch(opt)

    # Get interesting data
    df = read_dataset("data.csv")
    x, y = get_xy_from_dataframe(df)

    # Train model
    training(model, x, y)

# Open dataset with panda
def read_dataset(filename):
    df = pd.read_csv(filename)
    return df

# Drop useless columns and return x and y
def get_xy_from_dataframe(df):
    x = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
    y = df['class'].values
    return x, y 

# Ask for model choice
def prompt_display():
    print("""Choose a model:

(1) - KNN
(2) - Tree
(3) - RandomForestClassifier
(4) - SGD
(5) - Linear SVC""")
    return int(input())

def model_switch(choice):
    if (choice == 1):
        model = KNeighborsClassifier()
    elif (choice == 2):
        model = DecisionTreeClassifier(random_state=0, max_depth=20)
    elif (choice == 3):
        model = RandomForestClassifier(n_estimators=100 ,criterion='entropy')
    elif (choice == 4):
        model = sgdclassifier(max_iter=1000, tol=0.01)
    elif (choice == 5):
        model = svm.svc(kernel='linear', c = 1.0)    
    else:
        raise Exception('Wrong entry')       
    
    return model

def plot_columns_hist(columns):
    x.hist()
    plt.show()
    
def printPredictedValues(ypredit,ytest):
    for i in range(0,len(ypredit)):
        print("✅ Prédit/Réel: ",ypredit[i],ytest[i]) if ypredit[i]==ytest[i] else print("🔴 Prédit/Réel: ",ypredit[i], ytest[i])

def printStatValues(ypredit,ytest):
    galaxyStats = 0 
    starStats = 0 
    QSOStats = 0 
    N = len(ypredit)
    NF = 0
    for i in range(0,N):
        if ypredit[i] != ytest[i]:
            NF +=1
            if ypredit[i] == "GALAXY":
                galaxyStats+=1
            elif ypredit[i] == "QSO":
                QSOStats+=1 
            elif ypredit[i]=="STAR":
                starStats+=1 
    print("Répartition des prédiction fausses : ")
    print("Galaxy : ",(galaxyStats*100/NF),"%","Star :",(starStats*100/NF),"%","QSO : ",(QSOStats*100/NF),"%")

# Train model
def training(model, x, y,res=-1):
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    Xtrain = Xtrain.values
    Xtest = Xtest.values
    
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)
 
    model.fit(Xtrain,ytrain)
    
    ypredit = model.predict(Xtest)
    os.system("clear")
    while(res != 0):
        print(" Rentre un chiffre:\n\n1 - Stats %\n2 - Stats raw\n3 - accuracy_score")
        print("0 - QUIT")
        res = int(input())
        if(res == 1):
            os.system("clear")
            printStatValues(ypredit,ytest)
        elif(res == 2):
            os.system("clear")
            printPredictedValues(ypredit,ytest)
        elif res == 3:
            os.system("clear")
            print(accuracy_score(ytest, ypredit))
        elif res == 0:
            break
        else:
            raise Exception('Wrong entry')

def clearData(df):
    res = df["class"].value_counts()
    dtemp = df.sort_values(by=['class'])
    supr = int(res["GALAXY"]/1.5)
    dtemp.drop(dtemp.index[range(1,supr)])
    dtemp = dtemp.iloc[34000:]
    return dtemp

def showData(df):
    res = df["class"].value_counts()
    x = [res["GALAXY"],res["QSO"],res["STAR"]]
    plt.figure(figsize = (8, 8))
    plt.pie(x, labels = ['GALAXY', 'QSO', 'Star'])
    plt.legend()

def allModels(df):
    dfClone = df.copy()
    # Aditionnale model randomforestclassifier(n_estimators=100 ,criterion='entropy', n_jobs=-1)
    modelArray=  ['KNN','Classifier']
    dfTemp = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
    y = df['class'].values
    x = list(dfTemp.columns.values)
    datas = []
    for i in range(0,len(x)):
        arrayColumns = [x[i]]
        for j in range(i+1,len(x)):
            xValues = dfTemp[arrayColumns]
            for k in range(0,len(modelArray)):
                if modelArray[k] == "KNN":
                    model = model_switch(1)
                elif modelArray[k] == "Classifier":
                    model = model_switch(2)
                else:
                    model = model_switch(1)
                print("Model used : ",modelArray[k], "---- Case : ",model)
                print("X values used : ",arrayColumns)
                accu = customTrainingRaw(model,xValues,y,3)
                it = [modelArray[k],arrayColumns,accu]
                datas.append(it)
            arrayColumns.append(x[j])
    return datas

def customTrainingRaw(model, x, y,res=-1):
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    Xtrain = Xtrain.values
    Xtest = Xtest.values
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)
    model.fit(Xtrain,ytrain)
    ypredit = model.predict(Xtest)
    print(accuracy_score(ytest, ypredit))
    return accuracy_score(ytest, ypredit)

def bestModelFinder(datas):
    maxi = 0
    knnMean= 0
    treeMean= 0
    for i in range(0,len(datas)):
        if datas[i][0] == 'KNN':
            knnMean += datas[i][2]
        else:
            treeMean += datas[i][2]
        if (datas[i][2] > maxi):
            maxi = datas[i][2]
            res = datas[i]
    print("BEST CHOICE IS :", res)
    print("Knn mean accuracy_score : ", mean(knnMean))
    print("Knn variance accuracy_score : ", variance(knnMean))
    print("Knn ecart-type accuracy_score : ", stdev(knnMean))
    print("Tree mean accuracy_score : ", mean(treeMean))
    print("Tree variance accuracy_score : ", variance(treeMean))
    print("Tree ecart-type accuracy_score : ", stdev(treeMean))
   



df = read_dataset("../data.csv")
bestModelFinder(allModels(df))
