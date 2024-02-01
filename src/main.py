#!/usr/bin/python3
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn as sk
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Classification 


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

#charger les données


def training(df):
# alpha delta u g r i z redshift spec_OBJ_ID 
# Y : class 
    x = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
    y = df['class'].values

    print(" Rentre un chiffre:\n\n1 - KNN\n2 - Tree\n3- RandomForestClassifier")
    res = int(input())
    
    if(res == 1):
        model = KNeighborsClassifier()
    elif(res == 2):
        model = DecisionTreeClassifier(random_state=0, max_depth=20)
    elif(res == 3):
        model = RandomForestClassifier(n_estimators=100 ,criterion='entropy')
    else:
        raise Exception('Mauvaise saisie')
    
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    Xtrain = Xtrain.values
    Xtest = Xtest.values

    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)
    # Model training
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
            raise Exception('Mauvaise saisie')

def clearData(df):
    res = df["class"].value_counts()
    

df=pd.read_csv('../data.csv')
clearData(df)
