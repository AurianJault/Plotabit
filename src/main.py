#!/usr/bin/python3
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn as sk

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from sklearn.externals.joblib import parallel_backend

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn.metrics import confusion_matrix

# main
def main():
    # plotAll()
    # auto_sklearn()

    # dftmp = pd.read_csv('data.csv')
    # dftmp.info()
    # print(dftmp['class'].value_counts())
    
    # User input
    opt = prompt_display()
    model = model_switch(opt)

    # Get interesting data
    df, x, y = read_dataset('data.csv')

    # rfecv_test(x, y, RandomForestClassifier())
    # Train model
    training(model, x, y)

# Open dataset with panda
def read_dataset(filename):
    df = pd.read_csv(filename)
    x = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
    y = df['class'].values

    return df, x, y
    
# Ask for model choice
def prompt_display():
    print("""Choose a model:
(1) - KNN (auto)
(2) - KNN (ball_tree, n=5)
(3) - Tree
(4) - RandomForestClassifier
(5) - SGD
(6) - Linear SVC
(7) - NearestCentroid
(8) - MLPClassifier""")
    return int(input())

def model_switch(choice):
    if (choice == 1):
        model = KNeighborsClassifier(algorithm="auto")
    elif (choice == 2):
        model = KNeighborsClassifier(n_neighbors=2, algorithm="ball_tree", weights="distance")
    elif (choice == 3):
        model = DecisionTreeClassifier(random_state=0, max_depth=20)
    elif (choice == 4):
        model = RandomForestClassifier(n_estimators=100 ,criterion='entropy', n_jobs=-1)
    elif (choice == 5):
        model = SGDClassifier(max_iter=1000, tol=0.01)
    elif (choice == 6):
        model = LinearSVC(C=1.0, dual=False, verbose=True, loss="squared_hinge", multi_class="crammer_singer")
    elif (choice == 7):
        model = NearestCentroid()
    elif (choice == 8):
        model = MLPClassifier(solver='adam', alpha=1e-5, random_state=1, activation="logistic", hidden_layer_sizes=(1000, 300, 100, 30, 10, 3))
    else:
        raise Exception('Wrong entry')       
    
    return model

def plot_columns_hist(columns):
    x.hist()
    plt.show()


def printPredictedValues(ypredit,ytest):
    for i in range(0,len(ypredit)):
        print("✅ Prédit/Réel: ",ypredit[i],ytest[i]) if ypredit[i]==ytest[i] else print("🔴 Prédit/Réel: ",ypredit[i], ytest[i])

# Affiche le pourcentage d'apparition dans les résultats faux du modèle pour chacune des classes
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
def training(model, x, y):
    # Sépare les données test (25%) des données d'entrainement (65%)
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    Xtrain = Xtrain.values
    Xtest = Xtest.values
    
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)
    
    # Entrainement du model choisi
    model.fit(Xtrain,ytrain)

    ypredict = model.predict(Xtest)
    res = -1
    while(res != 0):
        print(" Rentre un chiffre:\n\n1 - Stats %\n2 - Stats raw\n3 - accuracy_score")
        print("0 - QUIT")
        res = int(input())
        if(res == 1):
            os.system("clear")
            printStatValues(ypredict,ytest)
        elif(res == 2):
            os.system("clear")
            printPredictedValues(ypredict,ytest)
        elif res == 3:
            os.system("clear")
            print(accuracy_score(ytest, ypredict))
        elif res == 0:
            break
        else:
            raise Exception('Wrong entry')


# Divise par 1.5 le nombre de galaxy 
# (On a essayé d'équilibrer les différentes classes mais ça a affaiblit nos modèles à chaque fois)
def clearData(df):
    res = df["class"].value_counts()
    dtemp = df.sort_values(by=['class'])
    supr = int(res["GALAXY"]/1.5)
    dtemp.drop(dtemp.index[range(1,supr)])
    dtemp = dtemp.iloc[34000:]
    return dtemp

# Affiche un camembert de la répartition des classes présentes dans le csv
def showData(df):
    res = df["class"].value_counts()
    x = [res["GALAXY"],res["QSO"],res["STAR"]]
    plt.figure(figsize = (8, 8))
    plt.pie(x, labels = ['GALAXY', 'QSO', 'Star'])
    plt.legend()

def rfecv_test(x, y, model):
    rfe = RFECV(estimator=model)
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise', verbose=3)
    
    # report performance
    print('Accuracy: %.3f (%.3f)' % (max(n_scores), std(n_scores)))

    rfe.fit(x,y)
    for i in range(x.shape[1]):
        print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))

# Entraine "tout" les modèles en itérant sur les colonnes fournis (pas tous les cas car trop long).
# Seulement sur les modèles KNN et TreeClassifier car trop long
# EXEMPLE => colonnes = [A,B,C]
# itèrations :
# [A], [A,B], [A,B,C]
# [B], [B,C]
# [C]
def allModels(df):
    dfClone = df.copy()
    # Aditionnale model randomforestclassifier(n_estimators=100 ,criterion='entropy', n_jobs=-1)
    modelArray=  ['KNN','Classifier']
    dfTemp = df.drop(['obj_ID','field_ID','run_ID','rerun_ID','cam_col','plate','MJD','fiber_ID','class'],axis=1)
    y = df['class'].values
    x = list(dfTemp.columns.values)
    datas = []
    # Itère sur les colonnes
    for i in range(0,len(x)):
        arrayColumns = [x[i]]
        for j in range(i+1,len(x)):
            xValues = dfTemp[arrayColumns]
            
            # Knn model train
            model = model_switch(1)
            accuKnn = customTrainingRaw(model,xValues,y,3)
            print("Model used : Knn ---- Case : ",model)
            print("X values used : ",arrayColumns)

            # Tree model train
            model = model_switch(3)
            accuTree = customTrainingRaw(model,xValues,y,3)
            print("Model used : Tree ---- Case : ",model)
            print("X values used : ",arrayColumns)

            
            dico = dict()
            setUp = [arrayColumns.copy(),dico]
            setUp[1]['Knn'] = accuKnn
            setUp[1]['Tree'] = accuTree
            datas.append(setUp.copy())
            
            arrayColumns.append(x[j])
    # datas est de la forme suivante :
    # datas =  
    # [
    #   [Listes_des_colonnes],
    #   {'KNN': acccuray,'Tree': accuracy}
    # ]
    return datas

# Permet d'entrainer un model et retourne l'accuracy score
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

# Créer un nuiage de points de l'accuracy en fonction des colonnes itérées pour Knn et Tree
def showStat(datas):
    fig, ax = plt.subplots()
    x_data = []
    y_dataKnn = []
    y_dataTree = []
    for i in range(0,len(datas)):
        x_data.append("/".join(datas[i][0]))
        y_dataKnn.append(datas[i][1]['Knn'])
        y_dataTree.append(datas[i][1]['Tree'])

    ax.scatter(x_data, y_dataKnn, label=f'Y = Knn')
    ax.scatter(x_data, y_dataTree, label=f'Y = Tree')
    ax.set_xlabel('Axe X')
    ax.set_ylabel('Axe Y')
    ax.legend()
    plt.show()

# Trouve la meilleur accuracy et print le model + les colonnes
def bestModel(datas):
    max = 0
    min = 1
    for i in range(0,len(datas)):
        if(datas[i][1]['Knn'] < min):
            min = datas[i][1]['Knn']
            resMin = datas[i]
            modelMin = 'Knn'
        elif datas[i][1]['Tree'] < min:
            min = datas[i][1]['Tree']
            resMin = datas[i]
            modelMin = 'Tree'

        if(datas[i][1]['Knn'] > max):
            max = datas[i][1]['Knn']
            res = datas[i]
            model = 'Knn'
        elif datas[i][1]['Tree'] > max:
            max = datas[i][1]['Tree']
            res = datas[i]
            model = 'Tree'
    print("Best model : ",model," columns : ",res[0]," Accuracy : ", res[1][model])
    print("Worst model : ",modelMin," columns : ",resMin[0]," Accuracy : ", resMin[1][model])

def auto_sklearn():
    df = read_dataset('data.csv')
    X_train, X_test, y_train, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    X_train = X_train.values
    X_test = X_test.values
    
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)
        
    cls = autosklearn.classification.AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    y_hat = predictions = cls.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

def plotAll():
    x,df,y = read_dataset('data.csv')
    
    plotHistograms(df)
    plotDensity(df)
    plotBoxWhisker(df)
    plotCorrelationMatrix(df)
    plotScatterMatrix(df)

def plotHistograms(df):
    df.hist()
    plt.show()

def plotDensity(df):
    df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
    plt.show()

def plotBoxWhisker(df):
    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
    plt.show()

def plotCorrelationMatrix(df):
    correlations = df.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,9,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(list(df))
    ax.set_yticklabels(list(df))
    plt.show()

def plotScatterMatrix(df):
    scatter_matrix(df)
    plt.show()

# Affiche la répartitions des objets stélaires dans la base de données
#showData(df)

# Affiche le meilleur models avec les meilleurs colonnes entre KNeighborsClassifier et DecisionTreeClassifier
#datas = allModels(df)
#bestModel(datas)

# Génère un nuage de points affichant l'accuracy du model Knn et TreeClassifier en fonction des colonnes utilisées.
# datas = allModels(df)
# showStat(datas)
# bestModel(datas)

# Affiche un menu permettant de choisir le model à entrainer, ainsi que des stats suplémentaires
main()
