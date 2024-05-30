# Project 1 - Term Deposit Marketing


## Introduction
This project pertains to the Finance and Banking industry. A small startup catering to the European Banking Market was looking to optimize its success rate with one of its marketing campaigns. 

The company wanted to build a robust machine-learning system from data coming from a call center via the marketing effort of a European banking institution. 

The goal was to analyze data, design, and implement a machine-learning model that would solve the following:

1) Predict if the customer will subscribe (yes/no) to a term deposit
2) Which customers are likely to buy the investment product? Determine the segments that the firm must prioritize
3) What makes the customers buy? Which feature the firm should focus more on?

## Methodology
Project Overview:<br>
The following steps were taken to achieve the results for the goals outlined above:

1) Null Value Check<br>
   Performed checks for null values and distribution. Further EDA on the whole dataset was restricted due to MLOps approach. However, univariate, bivariate and other analyses were performed on the training dataset.
   
2) Train Test Split<br>
   The dataset was divided into 3 subsets namely train, validation, and test set. The composition was as follows:<br>
     - Train set made up ~80%<br>
     - Validation set made up ~10%<br>
     - Test set made up ~10%<br>
       
3) EDA<br>
   The train set's features, distribution, and outliers were analyzed. Robust Scaler was adopted for features whose distribution lay outside the interquartile range. In addition, all categorical and continuous variables were identified and segmented for different encoding techniques. 

4) Encoding<br>
   Three encoding techniques were implemented across the training dataset namely Label, Ordinal, and One-hot Encoding. The same 3 techniques were later applied to validation and test sets. 

5) Feature Engineering<br>
   Correlations were explored across the training dataset. There was only 1 feature (Duration) that was relatively highly correlated to the target variable. Thus steps were taken to reduce the feature size that could explain most of the variability without any loss of information. Recursive Feature Engineering (RFE) was employed where f1 scores were measured after the subsequent addition of individual features. I found 12/27 features that were relevant to our problem.

 6) Class Imbalance Check<br>
    Given the nature of the highly imbalanced dataset, a combination of oversampling and undersampling was implemented on the training set. I used the SMOTE algorithm to oversample the minority class and Random_under_sampler to undersample the majority class. Baseline model was measured before and after sampling which reported a 25 ppt improvement.
    
 7) Model Building & Grid Search<br>
    Model building was done using spot-checking framework. The following models were employed:<br>
    - Various Tree models such as Random Forest, light Gradient Boosting, Extreme Gradient Boosting, and extra tree were employed<br>
    - logistic regression<br>
    - K- Nearest Neighbour<br>
    - Naive Bayes Classifier<br>
   
    -Gridsearch was performed on the best-performing model to fine-tune the hyperparameters  
   
  8) Evaluation<br>
     5 fold cross-validation was performed on the final train set with F1 score as the performance metric.

   9) Results<br>
      Fine-tuned model was deployed on cross-validation and test sets whose performance was reported.


## Conclusion:

The model was able to predict with great accuracy, whether the customer would subscribe to a term deposit or not, with a final f1 score of 0.97 and 5 fold-cross validation. Following were the steps taken to achieve a final f1 score of 0.97 with 5 fold-cross validation:
1) Reduced data leakage
2) Scaled different features appropriately
3) Used a recursive feature engineering technique (RFE) to filter 27 features
4) Used a combination of SMOTE and undersampling to balance the class
5) Applied Spot-check framework 
6) Fine-tuned and performed 5 fold cross validation

Duration was one of the most important features that led to conversion. The company must focus on delivering quality sales pitch and engagement to further enhance customer experience. Other features that the company can potentially use to better target their customers for conversions were found as follows:

Features: 'education', 'housing', 'month', 'job_blue-collar', 'job_management', 'marital_married', 'contact_cellular', 'age', 'balance', 'day','campaign'
(Please note that the above features only enable to prioritize purchasing customers)



    






