# Plotabit
Dataset link: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

Deep Learning: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

Python Machine Learning: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

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

- [ ] Train models with just 20000 "GALAXY" class (has an impact?)
- [ ] Which model is the best, ratio learn_time/precision
- [ ] Can we drop more categories and have same results (useless data?)
- [ ] Compare prediction with y_test that were false
