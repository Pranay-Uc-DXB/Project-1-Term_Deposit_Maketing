# Project 1 - Term Deposit Marketing

This project pertains to the Finance and Banking industry. A small startup catering to the European Banking Market was looking to optimize its success rate with one of its marketing campaigns. 

The company wanted to build a robust machine-learning system from data coming from a call center via the marketing effort of a European banking institution. 

The goal was to analyze data, design, and implement a machine-learning model that would solve the following:

1) Predict if the customer will subscribe (yes/no) to a term deposit
2) Which customers are likely to buy the investment product? Determine the segments that the firm must prioritize
3) What makes the customers buy? Which feature the firm should focus more on?


Project Overview:
The following steps were taken to achieve the results for the goals outlined above:

1) Null Value Check
   Performed checks for null values and distribution. Further EDA on the whole dataset was restricted due to MLOps approach. However, univariate, bivariate and other analyses were performed on the training dataset.
   
2) Train Test Split
   The dataset was divided into 3 subsets namely train, validation, and test set. The composition was as follows:
     - Train set made up ~80%
     - Validation set made up ~10%
     - Test set made up ~10%
       
3) EDA
   The train set's features, distribution, and outliers were analyzed. Robust Scaler was adopted for features whose distribution lay outside the interquartile range. In addition, all categorical and continuous variables were identified and segmented for different encoding techniques. 

4) Encoding
   Three encoding techniques were implemented across the training dataset namely Label, Ordinal, and One-hot Encoding. The same 3 techniques were later applied to validation and test sets. 

5) Feature Engineering
   Correlations were explored across the training dataset. There was only 1 feature (Duration) that was relatively highly correlated to the target variable. Thus steps were taken to reduce the feature size that could explain most of the variability without any loss of information. Recursive Feature Engineering (RFE) was employed where f1 scores were measured after the subsequent addition of individual features. I found 12/27 features that were relevant to our problem.

 6) Class Imbalance Check
    Given the nature of the highly imbalanced dataset, a combination of oversampling and undersampling was implemented on the training set. I used the SMOTE algorithm to oversample the minority class and Random_under_sampler to undersample the majority class. Baseline model was measured before and after sampling which reported a 25 ppt improvement.
    
 7) Model Building & Grid Search
    Model building was done using spot-checking framework. The following models were employed:
    - Various Tree models such as Random Forest, light Gradient Boosting, Extreme Gradient Boosting, and extra tree were employed
    - logistic regression
    - K- Nearest Neighbour
    - Naive Bayes Classifier
   
    -Gridsearch was performed on the best-performing model to fine-tune the hyperparameters  
   
  8) Evaluation
     5 fold cross-validation was performed on the final train set with F1 score as the performance metric.

   9) Results
      Fine-tuned model was deployed on cross-validation and test sets whose performance was reported.


Conclusion:


    






