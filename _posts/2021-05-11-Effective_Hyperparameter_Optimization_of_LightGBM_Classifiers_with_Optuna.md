
# Effective Hyperparameter Optimization of LightGBM Classifiers with Optuna

LightGBM is a really convenient to use, fast to train and usually accurate implementation of boosted trees. Here I use optuna for hyperparameter search using Bayesian optimization methods, with 5-fold cross validation, to gain a fairly accurate model. Also I leverage some seemingly minor but very useful built in features of the LightGBM library to handle categorical variables.

Note that the dataset I am using for this blog is the kaggle tabular playground march-2021 dataset which could be downloaded at https://www.kaggle.com/c/tabular-playground-series-mar-2021. Also note that I am not going to go over the EDA aspects of the problem because the purpose is to show the ease with which optuna could be used to tune the hyperparameters of LightGBM to yield highly accurate models.


```python
##Import the required packages. These include pandas, numpy,scikit-learn and optuna

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
import optuna
import optuna.integration.lightgbm as lgb


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


<style type='text/css'>
.datatable table.frame { margin-bottom: 0; }
.datatable table.frame thead { border-bottom: none; }
.datatable table.frame tr.coltypes td {  color: #FFFFFF;  line-height: 6px;  padding: 0 0.5em;}
.datatable .bool    { background: #DDDD99; }
.datatable .object  { background: #565656; }
.datatable .int     { background: #5D9E5D; }
.datatable .float   { background: #4040CC; }
.datatable .str     { background: #CC4040; }
.datatable .row_index {  background: var(--jp-border-color3);  border-right: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  font-size: 9px;}
.datatable .frame tr.coltypes .row_index {  background: var(--jp-border-color0);}
.datatable th:nth-child(2) { padding-left: 12px; }
.datatable .hellipsis {  color: var(--jp-cell-editor-border-color);}
.datatable .vellipsis {  background: var(--jp-layout-color0);  color: var(--jp-cell-editor-border-color);}
.datatable .na {  color: var(--jp-cell-editor-border-color);  font-size: 80%;}
.datatable .footer { font-size: 9px; }
.datatable .frame_dimensions {  background: var(--jp-border-color3);  border-top: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  display: inline-block;  opacity: 0.6;  padding: 1px 10px 1px 5px;}
</style>



    /kaggle/input/tabular-playground-series-mar-2021/sample_submission.csv
    /kaggle/input/tabular-playground-series-mar-2021/train.csv
    /kaggle/input/tabular-playground-series-mar-2021/test.csv


Read the train and test csv's into variables and list out the column names


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.columns.to_list()
```




    ['id',
     'cat0',
     'cat1',
     'cat2',
     'cat3',
     'cat4',
     'cat5',
     'cat6',
     'cat7',
     'cat8',
     'cat9',
     'cat10',
     'cat11',
     'cat12',
     'cat13',
     'cat14',
     'cat15',
     'cat16',
     'cat17',
     'cat18',
     'cont0',
     'cont1',
     'cont2',
     'cont3',
     'cont4',
     'cont5',
     'cont6',
     'cont7',
     'cont8',
     'cont9',
     'cont10',
     'target']



Convert the categorical data into the category type such that lightgbm can handle the categorical variables. Unless you leverage learned embeddings for categorical variables, this fares better that one hot encoding or label encoding


```python
conts = ['cont0','cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10']
cats = ['cat0','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18','target']
for c in train.columns:
    col_type = train[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        train[c] = train[c].astype('category')

for c in test.columns:
    col_type = test[c].dtype
    if col_type == 'object' or col_type.name == 'category':
        test[c] = test[c].astype('category')
```

Specify dependent and independent variables and create a lgb Dataset object


```python
X= train[['cont0','cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cat0','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18']]
Y = train[['target']]
```

Use the optuna lightGBM integration to do hyperparamater optimization with 5 fold cross validation. Make sure to pass in the argument 'auto' for categorical_feature for automated feature engineering for categorical input features.


```python
from sklearn.model_selection import StratifiedKFold
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Notice how I've specified auc as the metric.


```python
dtrain = lgb.Dataset(X,Y,categorical_feature = 'auto')

params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "boosting_type": "gbdt",
}

tuner = lgb.LightGBMTunerCV(
    params, dtrain, verbose_eval=100, early_stopping_rounds=1000000, folds=kfolds
)

tuner.run()

print("Best score:", tuner.best_score)
best_params = tuner.best_params
print("Best params:", best_params)
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))
```


```python
#Results
Best score: 0.8950986360077963
Best params: {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt', 'feature_pre_filter': False, 'lambda_l1': 7.959176411127531, 'lambda_l2': 5.381687699546818e-05, 'num_leaves': 135, 'feature_fraction': 0.4, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20}
  Params: 
    objective: binary
    metric: auc
    verbosity: -1
    boosting_type: gbdt
    feature_pre_filter: False
    lambda_l1: 7.959176411127531
    lambda_l2: 5.381687699546818e-05
    num_leaves: 135
    feature_fraction: 0.4
    bagging_fraction: 1.0
    bagging_freq: 0
    min_child_samples: 20
```

Inspect the best score for AUC value


```python
tuner.best_score
```




    0.8950986360077963



Assign the best params to a variable


```python
params = tuner.best_params
```

Use these parameters to train a LightGBM model on the entire training dataset 



```python
import lightgbm as lgb
id_test = test.id.to_list()
model = lgb.train(params, dtrain, num_boost_round=1000)
```

Predict on the test set and save file. Make sure you set index=False 


```python
X_test = test[['cont0','cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9','cont10','cat0','cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8','cat9','cat10','cat11','cat12','cat13','cat14','cat15','cat16','cat17','cat18']]
preds = model.predict(X_test)
resultf = pd.DataFrame()
resultf['id'] = id_test
resultf['target'] = preds
resultf.to_csv('submission.csv',index=False)
```


