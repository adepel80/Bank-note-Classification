#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os


# ## Data Exploration

# In[2]:


pwd


# In[3]:


columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'Target']
bank = pd.read_csv('bank.csv', header = None, names = columns)


# In[4]:


bank


# In[5]:


bank.info()


# In[6]:


bank.describe()


# In[7]:


#Naming all Numerical variables for easy analysis
num_cols = bank.select_dtypes(include=np.number).columns.tolist()
print("Numerical Variables:")
print(num_cols)


# # iii Visualizing the distribution of each feature and the class balance.

# In[8]:


# checking for skewness and outliers in our numeric variable 

for col in num_cols:
    print(col)
    print('Skew :', round(bank[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    bank[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=bank[col])
    plt.show()


# ### DATA PREPROCESSING

# In[9]:


#checking for null value
bank.isnull().sum()


# In[10]:


y = bank["Target"]
x = bank.drop (["Target"], axis = 1)


# ### Splitting of data to 70/30

# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[12]:


x_train.shape


# In[13]:


x_test.shape


# In[14]:


y_test.shape


# In[15]:


y_train.shape


# ## Scale the feature values

# In[16]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(x_train)
X_test = std.transform(x_test)


# # MODEL DEVELOPMENT
# ### LOGISTIC REGRESSION

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


LGR = LogisticRegression(C=2,
                        n_jobs=5,
                        random_state=2)


# In[19]:


LGR.fit(x_train, y_train)


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the validation set
y_pred_lr = LGR.predict(x_test)

# Evaluate the model
print("Logistic Regression Evaluation:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))


# In[21]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", accuracy)


# In[22]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
y_pred_lr = LGR.predict(x_test)
cm = confusion_matrix(y_test, y_pred_lr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LGR.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# ## RANDOM FOREST

# In[23]:


from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')


# In[24]:


rf = RandomForestClassifier(n_estimators=20)


# In[25]:


rf.fit(x_train, y_train)


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the validation set
y_pred = rf.predict(x_test)

# Evaluate the model
print("Logistic Regression Evaluation:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[27]:


#evaluae the model
y_pred = rf.predict(x_test)
accuracy_score(y_test, y_pred)


# In[28]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# # MODEL EVALUATION
# 
# # (i) Performance of models using a confusion matrix.
# 
# A confusion matrix is a table used to describe the performance of a classification model on a set of test data for which the true values are known. It helps in understanding the types of errors the model is making. Here's the typical arrangement of the confusion matrix along with the definitions of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN):
# 
# ### 1 Logistic regression
# 
# True Positive (TP): 226
# 
# True Negative(TN): 2
# 
# False Positive(FP): 183
# 
# False Negative(Fn): 3
# 
#  ### 2 Random forest
# 
# True Positive (TP): 229
# 
# True Negative(TN): 2
# 
# False Positive(FP): 183
# 
# False Negative(Fn): 3
# 
# 
# #### True Positives (TP) Algorithm correctly predicted authentic banknote
# 
# #### True Negatives (TN) Algorithm correctly predicted counterfeit bank note, 
# 
# #### False Positives (FP) Algorithm predicted authentic bank note, but its counterfeit
# 
# #### False Negatives (FN) Algorithm predicted counterfeit bank note, but its Authentic
# 
# 
# # Conclusion
# 
# Based on the provided TP values, the random forest seems to have a slight edge.
# 
# 
# 
# ## (ii) STRENGTH AND WEAKNESS OF MODEL
# 
# ### 1 Logistic Regression
# 
# ### Strengths:
# 
# Interpretable coefficients.
# 
# Computationally efficient.
# 
# Provides probabilistic outputs.
# 
# ### Weaknesses:
# 
# Assumes linear relationships.
# 
# Sensitive to outliers.
# 
# ### 2 Random Forest
# 
# ### Strengths:
# 
# Captures non-linear relationships.
# 
# High performance and robustness.
# 
# ### Weaknesses:
# 
# Less interpretable.
# 
# More computationally intensive
# 
# ### Performance Metrics
# 
# Logistic Regression: Have lower accuracy and same F1-score on complex, non-linear data, but offers clear interpretation and efficiency.
# 
# Random Forest: have higher accuracy, same precision and recall, and ROC-AUC on complex datasets but at the cost of interpretability
# and computational resources.
# 

# # MODEL OPTIMIZATION USING THE GRIDSEARCHCV

# In[29]:


from sklearn.model_selection import train_test_split, GridSearchCV
model = RandomForestClassifier(random_state=42)


# In[30]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# In[31]:


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)


# In[32]:


best_model = grid_search.best_estimator_


# In[33]:


y_pred = best_model.predict(x_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# # CONCLUSION
# 
# ##  (i) FINDINGS IN THE PERFORMANCCE OF THE RANDOM FOREST USING THE GRID SEARCH 
# 
# The Random Forest classifier outperformed Logistic Regression across all performance metrics, including accuracy (0.99), 
# precision (1), recall (0.99), and F1 score (1). These results indicate that Random Forest is more effective in
# classifying banknotes as authentic or counterfeit.

# # (ii) Insights into which model is more suitable for this task and why. 
# 
# Using grid search to optimize the Random Forest model significantly enhances its performance by fine-tuning hyperparameters like the number of trees and maximum depth. This optimization leads to better handling of complex data patterns and interactions compared to Logistic Regression. The Random Forest's ensemble nature and robustness to overfitting make it more suitable for accurately classifying banknotes as authentic or counterfeit. Its superior performance metrics, including accuracy, precision, recall, and F1 score, indicate its effectiveness and reliability for this task. Thus, the optimized Random Forest model is the more suitable choice.
# 
# # (iii)	Discuss any potential improvements or future work that could be done.
# 
# Using grid search techniques with the Random Forest model can significantly enhance performance by fine-tuning hyperparameters such as the number of trees, maximum depth, and minimum samples per leaf. Implementing feature engineering strategies, like creating new features and selecting the most relevant ones, can further boost model accuracy. Ensemble methods, such as stacking with other algorithms like XGBoost, can improve predictive power. Regularly retraining the model with new data ensures it remains effective over time. Monitoring and evaluating the model in real-world scenarios will help maintain its performance. Finally, incorporating additional relevant data sources and using interpretability tools like SHAP values can provide deeper insights and improve model reliability.

# # FEATURE IMPORTANCE

# In[40]:


#bank.Series(rf.feature_importance_, index=x.columns).nlargest(10)


# In[36]:


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# # Train a RandomForestClassifier

# In[41]:



clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x, y)


# # GET THE FEATURE IMPORTANCE

# In[46]:



importances = clf.feature_importances_
indices = importances.argsort()[::-1]


# # PLOT THE FEATURE IMPORTANCE 

# In[44]:



plt.figure()
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), [x.columns[i] for i in indices], rotation=90)
plt.show()


# # CROSS VALIDATION

# ## (i) Perform k-fold cross-validation on the best-performing model and report the results

# In[47]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[48]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)


# # DEFINE THE K-FOLD CROSS VALIDATION

# In[49]:


k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# # PERFORM THE CROSS VALIDATION

# In[50]:


# Cross-validation scores
scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')

# Print the results
print(f'Scores for each fold: {scores}')
print(f'Mean accuracy: {np.mean(scores)}')
print(f'Standard deviation: {np.std(scores)}')


#  # CROSS VALIDATION RESULT BREAKDOWN
#    ### Scores for Each Fold
# The scores for each fold represent the accuracy of the model on the test set for each of the 5 folds in the cross-validation process:
# 
# Fold 1: 0.99272727
# 
# Fold 2: 0.99272727
# 
# Fold 3: 0.99270073
# 
# Fold 4: 1.00000000
# 
# Fold 5: 0.99270073
# 
# These scores indicate that the model performs consistently well across different subsets of the data, with accuracy values very close to 1, which implies high performance.
# 
# ### Mean Accuracy
# The mean accuracy is the average of the scores across all folds:
# 
# Mean accuracy
# =
# 0.99272727
# +
# 0.99272727
# +
# 0.99270073
# +
# 1.00000000
# +
# 0.99270073
# 5
# =
# 0.9941712010617121
# 
# ### Mean accuracy= 
# 5
# 0.99272727+0.99272727+0.99270073+1.00000000+0.99270073
# ​
#  =0.9941712010617121
# 
# This value indicates the overall accuracy of the model across all folds. A mean accuracy of approximately 0.9942 suggests that the model correctly classifies approximately 99.42% of the instances on average.
# 
# ### Standard Deviation
# The standard deviation measures the amount of variation or dispersion of the scores from the mean accuracy:
# 
# Standard deviation
# =
# 0.0029144236428144807
# 
# Standard deviation=0.0029144236428144807
# 
# A low standard deviation (in this case, approximately 0.0029) indicates that the model's accuracy is consistently close to the mean accuracy across different folds, suggesting stable and reliable performance.
# 
# ### Interpretation
# 
# High Mean Accuracy: The model is performing very well, with an average accuracy of about 99.42%. This suggests that the model is highly effective in classifying the data correctly.
# Low Standard Deviation: The small standard deviation shows that the performance of the model is consistent across different folds of the data. This implies that the model generalizes well and is not overly dependent on any specific subset of the data.
# 
# ### Conclusion
# 
# The results indicate that the model has high and consistent accuracy for banknote authentication, making it a reliable choice for deployment in a real-world banking environment. The high mean accuracy and low standard deviation demonstrate both the effectiveness and stability of the model
# 

# # DEPLOYMENT CONSIDERATION

# ## (i) Discuss how to deploy a model in a real - world banking environment

# # 1. Model Selection and Training
# Select a Suitable Model: Choose a model that has been trained and validated on appropriate data for banknote authentication.
# For example, a Random Forest or Support Vector Machine (SVM) classifier might be suitable.
# # 2. Model Evaluation
# Evaluate Performance: Assess the model’s performance using evaluation metrics such as accuracy, precision, recall, and F1-score. Validate its ability to generalize to new, unseen data.
#  # 3. Data Pipeline Setup
# Data Integration: Develop a pipeline to seamlessly integrate new banknote data into the model. This might involve setting up data collection mechanisms or APIs to feed data into the system.
# Preprocessing and Featurization: Implement preprocessing steps to clean and prepare incoming data for the model. Ensure consistency in data formats and handling missing values.
#  # 4. Model Deployment
# Containerization: Package the model into a container (e.g., Docker) to ensure portability and reproducibility across different environments.
# API Development: Expose the model through an API (Application Programming Interface) for easy integration into banking systems. Consider security measures like authentication and encryption.
# Scalability: Ensure the deployed model can handle varying loads of authentication requests efficiently.
# # 5. Security and Compliance
# Data Privacy: Implement measures to protect sensitive data, ensuring compliance with regulations such as GDPR or local banking laws.
# Model Security: Secure the model and API against potential attacks, including input validation and monitoring for anomalies.
# Auditability: Maintain logs and audit trails of model predictions and system activities for transparency and accountability.
# # 6. Integration with Banking Systems
# Testing: Conduct thorough testing of the integrated system to ensure seamless operation and accuracy in real-world scenarios.
# Deployment Plan: Plan and execute the deployment in stages, considering fallback mechanisms and rollback procedures if issues arise.
# Training and Support: Provide training and support to banking personnel who interact with or rely on the model’s outputs.
# # 7. Monitoring and Maintenance
# Performance Monitoring: Continuously monitor the model’s performance and accuracy over time. Implement alerts for deviations from expected behavior.
# Feedback Loop: Establish a feedback mechanism to update the model periodically with new data and improve its accuracy.
# Maintenance: Regularly update dependencies and retrain the model as needed to ensure it remains effective and compliant with evolving regulations.
# # 8. Documentation and Reporting
# Documentation: Maintain comprehensive documentation covering model architecture, deployment steps, and operational procedures.
# Reporting: Generate periodic reports on model performance, security audits, and compliance status for stakeholders and regulatory bodies.
# By following these steps, you can effectively deploy a machine learning model for banknote authentication in a real-world banking environment, ensuring it meets operational requirements while adhering to security and compliance standards.
# 
# 
# 
# 
# 
# 
# 

# # (ii) Ethical concerns or potential biases in the model.

# # Ethical concerns or potential biases in the model
# When deploying a machine learning model it is crucial to address potential ethical concerns and biases to ensure fairness, reliability, and trustworthiness. Here are some key considerations:
# 
# ### 1. Data Bias
# 
# Representative Data: Ensure that the training data includes a diverse and representative sample of both authentic and counterfeit banknotes. If the data is skewed towards certain types of banknotes, the model may not perform well on underrepresented types.
# Source of Data: Verify the authenticity and quality of the data sources. Using data from unreliable sources may introduce biases or inaccuracies.
#     
# ### 2. Algorithmic Bias
# 
# Fairness: Assess the model for any inherent biases that could lead to unfair treatment of certain types of banknotes. For example, if the model is more accurate for newer banknotes than older ones, it might lead to unfair outcomes.
# Bias Detection: Implement methods to detect and mitigate biases in the model. Techniques like fairness-aware machine learning can help adjust the model to ensure balanced performance.
#     
# ### 3. Transparency and Explainability
# 
# Model Explainability: Ensure that the model’s decision-making process is transparent and explainable. Use techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to explain why a particular banknote is classified as authentic or counterfeit.
# Documentation: Maintain comprehensive documentation on how the model was developed, trained, and validated. This includes the data sources, preprocessing steps, feature selection, and algorithm choices.
#     
# ### 4. Security and Privacy
# 
# Data Privacy: Protect the privacy of any sensitive information used in the model training and prediction processes. Follow data protection regulations like GDPR to ensure compliance.
# Model Security: Safeguard the model against potential adversarial attacks that could manipulate its predictions. Implement robust security measures to protect the model and its deployment environment.
#     
# ### 5. Human Oversight
# 
# Human-in-the-Loop: Implement a system where human experts can review and verify the model’s predictions, especially in cases where the model’s confidence is low or the stakes are high.
# Continuous Monitoring: Regularly monitor the model’s performance and update it as necessary. This includes retraining the model with new data to adapt to changes and improve accuracy.
#     
# ### 6. Ethical Considerations
# 
# Impact on Stakeholders: Consider the impact of the model on various stakeholders, including customers and bank employees. Ensure that the deployment of the model does not lead to undue harm or inconvenience.
# Responsibility and Accountability: Define clear lines of responsibility and accountability for the model’s outcomes. Establish protocols for addressing errors or biases that may arise.
# 

# In[ ]:




