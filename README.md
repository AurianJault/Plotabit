# Plotabit
Dataset link: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

Deep Learning: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Python Machine Learning: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

AI Plot data: https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

## Columns

|Keep         |Skip        |
|:-----------:|:----------:|
|alpha        |obj_ID      |
|delta        |run_ID      |
|u            |rerun_ID    |
|g            |plate       |
|r            |MJD         |
|i            |fiber_ID    |
|z            |            |
|redshift     |            |

## Analysis

- [X] Train models with just 20000 "GALAXY" class (has an impact?)
- [ ] Which model is the best, ratio learn_time/precision
- [ ] Can we drop more categories and have same results (useless data?)
- [ ] Compare prediction with y_test that were false

## Dataset
Nous avons décidés de prendre un dataset sur le site Kaggle, il contient 100 000 lignes qui réprésentent
chacune un objet stellaire observé en lui attribuant plusieurs caractéristiques comme son inclinaison,
les longueurs d'ondes observées et autres valeurs scientifiques.
Chaque ligne est donc associée à une classe qui peut-être "QSO" un quasar, "Galaxy" ou "Star" une étoile.

Notre première étape à été de regarder le dataset pour savoir si certaines données sont manquantes.
En utilisant `df.info()` nous pouvons avoir certaines informations sur les données, il ne manque aucune valeur.

Nous pouvons maintenant regarder la répartition des classes, celle-ci est assez inégale avec ~60.000 Galaxie,
~21.000 étoiles et ~19000 quasar. Nous pouvons en déduire que les galaxies sont plus communes mais cela
pourrait-il avoir une incidence sur la précision de notre modèle ?

Après avoir testé avec un nombre égal de Galaxies, Etoiles et Quasars, les résultats sont moins bon qu'en utilisant 
le dataset de base. La précision n'est donc pas impactée par le grand nombre de galaxies.


## Choix des données d'entrainement
Pour entrainer des modèles nous devons d'abord diviser notre dataset en deux parties, les caractéristiques
de chaque objet (x) ainsi que la classe à laquelle il est associé (y). La valeur (x) ne représente pas
forcément toutes des caractéristiques pertinantes, nous avons d'abord du effectuer un tri.

Voici la liste des colonnes avec leurs descriptions:

1) **obj_ID** = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
2) **alpha** = Right Ascension angle (at J2000 epoch)
3) **delta** = Declination angle (at J2000 epoch)
4) **u** = Ultraviolet filter in the photometric system
5) **g** = Green filter in the photometric system
6) **r** = Red filter in the photometric system
7) **i** = Near Infrared filter in the photometric system
8) **z** = Infrared filter in the photometric system
9) **run_ID** = Run Number used to identify the specific scan
10) **rereun_ID** = Rerun Number to specify how the image was processed
11) **cam_col** = Camera column to identify the scanline within the run
12) **field_ID** = Field number to identify each field
13) **spec_obj_ID** = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
14) **class** = object class (galaxy, star or quasar object)
15) **redshift** = redshift value based on the increase in wavelength
16) **plate** = plate ID, identifies each plate in SDSS
17) **MJD** = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
18) **fiber_ID** = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation

Plusieurs colonnes n'ont enfait aucun rapport avec l'astre en lui-meme, mais sont plutôt des informations sur
l'équipement utilisé pendant les observations. Nous les avons donc enlevées de (x) car leurs présence ne fait que
baisser la précision des modèles entrainés.

## Entrainement
Nous avons testés plusieurs modèles au cours de ce projet dans le but de trouver celui qui est le plus précis
pour prédire la catégorie d'un astre grace à ses caractéristiques. Au début de manière aléatoire en jouant
avec les hyper-paramètres, puis nous avons utilisés [l'arbre de choix de sklearn](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html).
Nous voulons prédire une catégorie, il faut donc utiliser un *Classifier* et nous avons 100.000 lignes, vu que la conditions est "<100K" nous avons utilisés
les modèles pour < et > à 100.000 lignes:

- KNN
- Decision Tree
- Linear SVC
- Random Forest

Mais aussi d'autre modèles pour tester leurs efficacité:

- Multi-Layer Perceptron
- Nearest Centroid
- SGD

Pour entrainer et tester nos modèles, il nous faut faire plusieurs groupes de données, celles d'entrainement
**Xtrain** et **Xtest**, pour les test **Ytrain** et **Ytest**. Ces groupes ont été générés en utilisant
`train_test_split(x, y,test_size=0.25, random_state=0)`, *test_size* défini le % du dataset à utiliser pour les tests.
L'ajout du paramètre `stratified=y` qui doit garder une répartition égale dans les groupes de test et train n'a pas
augmenté la précision de nos modèles pour autant.

Maintenant il ne reste plus qu'à utiliser la méthode `fit(Xtrain, Ytrain)` des modèles en utilisant **Xtrain** et **Ytrain**.
Ensuite il nous faut tester le modèle en lui faisant prédire les classes des données **Ytrain** en utilisant
`predictions = model.predict(Xtest)` et enfin récupérer la précision du modèle en comparant nos prédictions
avec **Ytest**: `accuracy_score(Ytest, predictions).`

## Résultat
Le meilleur modèle pour notre dataset est d'après nos expérimentations le **RandomForestClassifier** avec une précision
de 98%.

Voici les résultats obtenu sur l'ensemble des modèles:

- KNN (70,724%)
- Decision Tree (96,82%)
- Linear SVC (n'a jamais fini)
- Random Forest (98.012%)
- Multi-Layer Perceptron (59.22%)
- Nearest Centroid (36.328%)
- SGD (18.972%)
