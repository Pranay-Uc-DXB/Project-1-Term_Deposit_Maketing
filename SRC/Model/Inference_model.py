# %%
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
out = "..\Preprocessing steps.py"

x_sm= pd.read_pickle(out) 
y_sm= pd.read_pickle(out) 
x_test= pd.read_pickle(out) 
y_test= pd.read_pickle(out) 
x_val= pd.read_pickle(out) 
y_val=pd.read_pickle(out) 

# %%
import pickle
with open('Chosen_model.pkl','rb') as file:
    Script=pickle.load(file)
                       
Script  

# %%
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'

#Code to ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)

# %%
x_sm.shape
y_sm.shape
x_test.shape
y_test.shape
x_val.shape
y_val.shape

# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %%
def Predictor(model, x_data, y_data):
    model.fit(x_data,y_data)
    y_pred=model.predict(x_data)
    class_report=print(classification_report(y_data,y_pred))
    cm=confusion_matrix(y_data,y_pred)
    cm=sns.heatmap(cm,annot=True)
    return class_report, cm


# %%
# Performing test on validaiton test 

Predictor(Script,x_val,y_val)

# %%
# Performing test on test_set


Predictor(Script,x_test,y_test)

# %%
x_val.shape


# %%
x_test.shape

# %%



