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
Nous avons décidé de prendre un dataset sur le site Kaggle, il contient 100 000 lignes qui réprésentent
chacune un objet stellaire observé en lui attribuant plusieurs caractéristiques comme sa declinaison,
les couleurs observées et autres valeurs scientifiques.
Chaque ligne est donc associée à une classe qui peut-être "QSO" un quasar, "Galaxy" ou "Star".

Notre première étape à été de regarder le dataset pour savoir si certaines données sont manquantes.
En utilisant `df.info()` nous pouvons avoir certaines informations sur les données, il ne manque aucune valeur.

Nous pouvons maintenant regarder la répartition des classes, celle-ci est assez inégale avec ~60.000 Galaxie,
~21.000 étoiles et ~19000 quasar. Nous pouvons en déduire que les galaxies sont plus communes mais cela
pourrait-il avoir une incidence sur la précision de notre modèle ?

## Plot
J'ai la flemme d'analyser les plots que j'ai fait.
