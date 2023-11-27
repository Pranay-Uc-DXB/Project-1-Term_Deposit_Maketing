
# %%
# Saving the preprocessed data 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

out = "..\Preprocessing steps.py"

x_sm= pd.read_pickle(out) 
y_sm= pd.read_pickle(out) 
x_test= pd.read_pickle(out) 
y_test= pd.read_pickle(out) 
x_val= pd.read_pickle(out) 
y_val=pd.read_pickle(out) 

# %% [markdown]
# # Model Building

# %%
# Importing classification models

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# %%
## Preparing list of models to try out in Spot-Checking process

# We will use this user defined function to select the base model and then further optimize the model that yileded the best results

# We will choose 2 extremes (21,1001) to check model performance across range and then sunject best model to gridsearch

def model_zoo(models_=dict()):

    # Tree models
    for n_tress in [21,1001]:
        models_['rf' + str(n_tress)]=RandomForestClassifier(n_estimators=n_tress, n_jobs=-1,criterion='entropy')

        models_['lgb' + str(n_tress)]=LGBMClassifier(boosting_type='dart',n_jobs=-1,importance_type='gain')

        models_['xgb' + str(n_tress)]=XGBClassifier(n_estimators=n_tress, n_jobs=-1, criterion='entropy')

        models_['xtra' + str(n_tress)]=ExtraTreesClassifier(n_estimators=n_tress,criterion='entropy',n_jobs=-1)

    # Logistic Model
    models_['Log_reg']=LogisticRegression(penalty='l2',n_jobs=-1)
                                               
    # KNN Model
    for n in [3,5,7,11]:
        models_['KNN' + str(n)]=KNeighborsClassifier(n_neighbors=n)


    # Naive-Bayes models
    models_['gauss_NB']=GaussianNB()
    # models_['Multinomial_NB']=MultinomialNB()
    # models_['Compl_NB']= ComplementNB()
    models_['bern_nb']=BernoulliNB()

    return models_    

# %%
# Running and evaluating all models using KFold cross-validation (5 folds)

def evaluate_models(x_sm,y_sm, models_, folds=5, metric='f1'):
    results=dict()

    for name, model in models_.items():
    
        scores=cross_val_score(model,x_sm,y_sm,cv=folds,scoring=metric,n_jobs=1)

        #Scoring results of the evaluated model
        results[name]=scores
        mu, sigma=np.mean(scores), np.std(scores)

        #Printing individual model results
        print('Model {}: mean= {}, std_dev= {}'. format(name, mu, sigma))

    return results    

# %%
models_=model_zoo()
results=evaluate_models(x_sm,y_sm, models_, folds=5, metric='f1_macro')

# %%
results

# %% [markdown]
# From above we see extra-tree and xgboost model returned the best reults. Thus I will proceed to Hyper-parameter tuning and then fit the tuned model over our validation set. 

# %%
from sklearn.model_selection import GridSearchCV

# %%
# Selecting the best model from above and subjecting it to Gridsearch. 

# from above we see Extra tree classifier was the best performing model followed by xgboost
 
# Building list of parameters

params= {    'n_estimators':[50,100,200,300],
             'max_depth': [16,32,50,100,150],
             'min_samples_split':[2],
             'min_samples_leaf':[1],
             
}    


#'n_estimators':[int(x) for x in np.arange(start=10,stop=300,step=50)],
# 'max_depth': [int(x) for x in np.arange(start=10,stop=50,step=5)],


# %%
xtc=ExtraTreesClassifier(ExtraTreesClassifier(criterion='entropy',n_jobs=-1))

# %%
grid= GridSearchCV(xtc, param_grid=params, cv=5, scoring='f1', n_jobs=-1)
grid.fit(x_sm,y_sm)

# %%
grid.best_params_
grid.best_score_

# %%
import pickle

with open('Chosen_model.pkl','wb') as file:
    pickle.dump(grid,file)
# %%




# %%



