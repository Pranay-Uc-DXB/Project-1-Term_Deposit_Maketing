# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
#Code to get multiple outputs in the same cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'

#Code to ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)

# %%
df=pd.read_csv(r"C:\Users\97158\Desktop\Apziva\Project 2-Term Deposit Marketing\term-deposit-marketing-2020.csv")
df.head()

# %%
df.describe()
df.describe(include='O')

# %%
df['education'].unique()
df['job'].unique()
df['contact'].unique()

# %%
df.isnull().sum()

# %%
df_T=df.groupby(['y']).agg({'campaign':'count','duration':'mean','balance':'mean','age':'mean'}).reset_index().sort_values(by='balance')
df_T

# %% [markdown]
# From above it seems like a lot less time was spent with customers that said no despite really high number of campaigns compared to customers that said yes. Indicating correlation between y label and duration. 

# %%
# Seperating Target/Numerical/Categorical variables

y=df['y']
x=df

print('x Set Shape:', x.shape)
print('y Set Shape:', y.shape)


# %% [markdown]
# Seperating out train-test-validation sets
# 
# Since this is the only data available to us, we keep aside a holdout/test set to evaluate our model at the very end in order to estimate our chosen model's performance on unseen data/new data. 
# 
# A validatoin set will also be created as a basline model and to evaluate and tune our model(s).
# 
# We are performing this aplit here inorder to reduce potential data leakage

# %%
from sklearn.model_selection import train_test_split

# %%
#Train/Test Split
x_train_val, x_test, y_train_val, y_test=train_test_split(x,y,test_size=0.1,random_state=42)


#Train/Validation Split
x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_train_val, test_size=0.12, random_state=42)

# %%
x_train.head()

# %%
print('X_train Set: ',x_train.shape,' / ','Y_Train Set: ',y_train.shape)
print('X_Test Set: ', x_test.shape,' / ', 'Y_Test Set: ',y_test.shape)
print('X_CV Set: ',x_val.shape,' / ','Y_CV Set: ',y_val.shape)



# %% [markdown]
# Univariate Analysis - Performing a quick highlevel overvieew of outliers and distribution
# 
# 

# %%
#visulaizing distribution

x_train.hist(bins=5, color='steelblue', edgecolor='black',  grid=False)    
plt.tight_layout(rect=(0, 0, 1.2, 1.2))   

# %%
x_train[['age','balance','day','duration','campaign']].plot(kind='box',title='boxplot')

# %%
sns.displot(x_train['balance'])
sns.displot(x_train['duration'])

# %% [markdown]
# - From above we see that balance and duration are widely distributed and have some outliers
# 
# - Age and day show normal distribution
# 
# 

# %%
x_train['balance'].describe()

x_train['duration'].describe()


# %%
#Treating Outliers 

# Finding IQRs

percentile25=x_train['balance'].quantile(0.25)
percentile75=x_train['balance'].quantile(0.75)

percentile25_dur=x_train['duration'].quantile(0.25)
percentile75_dur=x_train['duration'].quantile(0.75)

# %%
iqr=percentile75-percentile25
iqr_dur=percentile75_dur-percentile25_dur


upper_limit= percentile75+1.5*iqr
lower_limit=percentile25-1.5*iqr

upper_limit_dur= percentile75_dur+1.5*iqr_dur
lower_limit_dur=percentile25_dur-1.5*iqr_dur


print('Balance max:', x_train['balance'].max(),'\n','Balance min:',x_train['balance'].min())
print('Balance Upper limit:',upper_limit,'\n','Balance lower limit',lower_limit)

# %%
outliers_upper=x_train[x_train['balance']>upper_limit]
print('% points outside upperlimit:',round(100*(outliers_upper.shape[0]/x_train.shape[0])),2,'%')

# %%
outliers_upper_dur=x_train[x_train['duration']>upper_limit_dur]
print('% points outside upperlimit:',round(100*(outliers_upper_dur.shape[0]/x_train.shape[0])),2,'%')

# %% [markdown]
# From above we see that the feature 'balance' is not Gaussian or Gaussian like. This prevents us from using Standard deviation method for outlier tratment. There are also too many records outside IQR to effectively ignore/exclude them from our analysis; thus we will not employ IQR method for outlier treatment. In addition, because capping involves  IQR's upper and lower limit, we will not be pursuing any outlier traatment. 

# %% [markdown]
# # Categorical Encoding

# %%
#Encoding categorical columns

print('Categorical columns and no. of unique values: ')
x_train.select_dtypes(include='O').nunique()

# %% [markdown]
# We have 9 categorical columns in the dataset including the target variable. 
# 
# Columns to undergo label encoding: <br>
# default <br>
# housing <br>
# loan <br>
# Target variable 'y' will be label encoded. 
# 
# Columns to undergo Ordinal Encoding: <br>
# education <br>
# month <br>
# 
# Columns to undergo One-Hot Encoding: <br>
# job <br>
# marital <br>
# contact

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# %%
label_col = ['default','housing','loan']
label_col_with_y = ['default','housing','loan','y']
Ordinal_col=['education','month']
Ohe_col=['job','marital','contact']

# %%
le=LabelEncoder()
oe=OrdinalEncoder()
ohe=OneHotEncoder(handle_unknown='ignore',sparse=False)

# Will be encoding only on train dataset as thats the only data we assume we have. We'll treat validation 
# and test sets as unseen data. Hence they cant be used for fitting the encoders. 

# %%
# Encoding train_set columns:

x_train[label_col] = x_train[label_col].apply(le.fit_transform)
y_train=le.fit_transform(y_train)
x_train[Ordinal_col] = oe.fit_transform(x_train[Ordinal_col])
x_train=pd.get_dummies(x_train,sparse=False,dtype=int)

# %%
# Applying the same across the test and validation set. 
x_test[label_col] = x_test[label_col].apply(le.fit_transform)
y_test=le.fit_transform(y_test)
x_test[Ordinal_col] = oe.fit_transform(x_test[Ordinal_col])
x_test=pd.get_dummies(x_test,columns=Ohe_col,dtype=int)

x_val[label_col] = x_val[label_col].apply(le.fit_transform)
y_val=le.fit_transform(y_val)
x_val[Ordinal_col] = oe.fit_transform(x_val[Ordinal_col])
x_val=pd.get_dummies(x_val,columns=Ohe_col,dtype=int)


# %%
x_val.shape
y_val.shape

# %% [markdown]
# # Feature Scaling

# %%
from sklearn.preprocessing import RobustScaler

# %%
cont_var=['age','balance','day','duration','campaign']

scaler=RobustScaler()

x_train_scaled=scaler.fit_transform(x_train[cont_var])

x_train_scaled=pd.DataFrame(x_train_scaled,columns=cont_var)
x_train_scaled.reset_index(drop=True, inplace=True)

# %%
x_train.drop(cont_var,axis=1,inplace=True)
x_train.reset_index(drop=True, inplace=True)
x_train=pd.concat([x_train,x_train_scaled],axis=1).reindex(x_train.index)

# %%
x_test_scaled=scaler.fit_transform(x_test[cont_var])
x_test_scaled=pd.DataFrame(x_test_scaled,columns=cont_var)
x_test_scaled.reset_index(drop=True, inplace=True) 

x_test.drop(cont_var,axis=1,inplace=True)
x_test.reset_index(drop=True, inplace=True)
x_test=pd.concat([x_test,x_test_scaled],axis=1).reindex(x_test.index)   



x_val_scaled=scaler.fit_transform(x_val[cont_var])
x_val_scaled=pd.DataFrame(x_val_scaled,columns=cont_var)
x_val_scaled.reset_index(drop=True, inplace=True) 

x_val.drop(cont_var,axis=1,inplace=True)
x_val.reset_index(drop=True, inplace=True)
x_val=pd.concat([x_val,x_val_scaled],axis=1).reindex(x_val.index)  


# %%
x_val.shape
y_val.shape

# %% [markdown]
# # Feature Engineering

# %%
plt.figure(figsize=(20,8))
corr=round(x_train.corr(),2)
sns.heatmap(corr,annot=True)
plt.show()

# %%
#Dropping Y label from train set


x_train.drop(['y_no','y_yes'], axis=1,inplace=True)
x_test.drop(['y'], axis=1,inplace=True)
x_val.drop(['y'], axis=1,inplace=True)

# %% [markdown]
# From above we see 'Duration' is the only feature that is showing correlation with Y labels. 

# %%
# Performing Recursive Feature Engineering
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from numpy import std
from numpy import mean



x_train.shape
y_train.shape

# %% [markdown]
# An important hyperparameter for the RFE algorithm is the 'number of features to select'. Because we do not know the best number of features to select, we will test out different values of features and see which feature/model returns better F1_score acorss the defined set of feature range.
# Below I demonstrate selecting different numbers of features from 2 to 28.

# %%
# Get a list of models to evaluate

def get_models():
    models=dict()
    for i in range(2,28):
        rfe=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=i)
        model=DecisionTreeClassifier()
        models[str(i)]=make_pipeline(rfe, model)
    return models

# %%
#Evaluating a given model model using Cross Validation

def evaluate_model(model, x_train, y_train):
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
    scores=cross_val_score(model,x_train,y_train,scoring='f1_macro', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# %%
# #getting the models to evaluate
# models=get_models()

# #evluating the models and storing results
# results,names=list(),list()
# for name, model in models.items():
#     scores=evaluate_model(model,x_train,y_train)
#     results.append(scores)
#     names.append(name)
#     print('>%s %.2f (%.2f)' % (name, mean(scores),std(scores)))

# #plot for model performance and comparison
# plt.boxplot(results, labels=names, showmeans=True)
# plt.show()

# %% [markdown]
# From above we see, after addition of 12 columns, the f1_Score of our model tapers off at 0.69. Thus we will be choosing 12 as our n_features_to_select. From here we will explore the top 12 columns that explain most of the patterns/behaviour/trends in our training dataset. 

# %%
#Defing RFE
rfe=RFE(estimator=DecisionTreeClassifier(),n_features_to_select=12)

#Fit RFE
rfe.fit(x_train,y_train)

#Summarizing all features
for i in range(x_train.shape[1]):
    print('Column: %d, Selected %s, RankL %.2f' %(i,rfe.support_[i],rfe.ranking_[i]))

# %% [markdown]
# From above we see Columns 0,2,4,6,9,18,20,23,24,25,26,27 are the minimun numbers of columns required to explain all the variation. We will now proceed to drop the remaining columns from all sets. 

# %%
x_train= x_train.iloc[:,[0,2,4,6,9,18,20,23,24,25,26,27]]
x_train.columns

# %%
x_test= x_test.loc[:,['education', 'housing', 'month', 'job_blue-collar', 'job_management',
       'marital_married', 'contact_cellular', 'age', 'balance', 'day',
       'duration', 'campaign']]
x_val= x_val.loc[:,['education', 'housing', 'month', 'job_blue-collar', 'job_management',
       'marital_married', 'contact_cellular', 'age', 'balance', 'day',
       'duration', 'campaign']]

# %% [markdown]
# # Class Imbalance Check 

# %%
unique_values, counts=np.unique(y,return_counts=True)

plt.bar(unique_values,counts)
plt.xlabel('Class label')
plt.ylabel('counts')
plt.title('Class Imbalance check')

# %% [markdown]
# From above we see that the class is imbalanced which is not healthy to be fed into any algorithim for predictions as this increases the probability of misclassifying minority class. We will thus use a combination of oversampling and undersampling techniques. We will couple SMOTE to oversample our minority class coupled with unsersampling of majority class to balance the Target label.

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from numpy import mean

# %%
#Defining our base model
model=DecisionTreeClassifier()

#Evaluating raw dataset
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
scores=cross_val_score(model,x_train,y_train,scoring='f1_macro', cv=cv,n_jobs=1)
print('Mean f1_Score: %.2f' % mean(scores))

# %%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline,Pipeline


# %%
#Defining pipeline
steps=[('over',SMOTE()),('model',DecisionTreeClassifier())]
pipeline=Pipeline(steps=steps)

#Evaluating the pipeline
scores=cross_val_score(pipeline,x_train,y_train,scoring='f1_macro', cv=cv,n_jobs=1)
print('Mean f1_Score: %.2f' % mean(scores))

# %%
# # Applying Decision tree on imbalalnced dataset with SMOTE and Random undersampling

# # values to evaluate
# k_val=[1,2,3,4,5,6,7]
# for k in k_val:
#    over=SMOTE(k_neighbors=k,random_state=2)
#    under=RandomUnderSampler(random_state=2)
#    steps=[('over',over),('under',under),('model',model)]
#    pipeline=make_pipeline(over,under,model)
#    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
#    scores=cross_val_score(pipeline,x_train,y_train,scoring='f1_macro', cv=cv,n_jobs=1)
#    score=mean(scores)
#    print('> k=%d, Mean f1_Score: %.2f' % (k,score))

# %%
# #model=DecisionTreeClassifier()
# # over=SMOTE(sampling_strategy=0.5,k_neighbors=3)
# under=RandomUnderSampler(sampling_strategy=1)
# pipeline=make_pipeline(over,under)
# x_sm,y_sm=pipeline.fit_resample(x_train,y_train)

# %%

model=DecisionTreeClassifier()
over=SMOTE(k_neighbors=4,random_state=2)
under=RandomUnderSampler()
pipeline=make_pipeline(over,under)

x_sm,y_sm=pipeline.fit_resample(x_train,y_train)

# %%
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

print("After OverSampling, the shape of train_X: {}".format(x_sm.shape))
print("After OverSampling, the shape of train_y: {} \n".format(y_sm.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_sm == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_sm == 0))) 


# %% [markdown]
# The dataset is transformed, first by oversampling the minority class followed by undersampling the majority class.

# %%
unique_values, counts=np.unique(y_sm,return_counts=True)

plt.bar(unique_values,counts)
plt.xlabel('Class label')
plt.ylabel('counts')
plt.title('Class Imbalance check')

# %%
from sklearn.metrics import f1_score

# %%
model=DecisionTreeClassifier()

#Evaluating the balanced dataset to see if there was improvement in baseline model after over and under sampling
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
scores=cross_val_score(model,x_sm,y_sm,scoring='f1_macro', cv=cv,n_jobs=1)
print('Mean f1_Score: %.2f' % mean(scores))

# %% [markdown]
# From above, we see that there was significant improvement of basline model after applying oversampling and undersammpling techniques. Now we can proceed to Model Building.

# %%
col_name=['y']
y_sm=pd.DataFrame(y_sm,columns=col_name)
y_test=pd.DataFrame(y_test,columns=col_name)
y_val=pd.DataFrame(y_val,columns=col_name)

# %%
import os,pickle

out = "..\Preprocessing steps.py"

x_sm.to_pickle(out)
y_sm.to_pickle(out)
x_val.to_pickle(out)
y_val.to_pickle(out)
x_test.to_pickle(out)
y_test.to_pickle(out)

# %%
