# Bank
Develop a machine learning model to accurately classify whether a banknote is authentic, or counterfeit based on various extracted features

# LIBRARIES
```
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
```

# DATA EXPLORATION
```
columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'Target']
bank = pd.read_csv('bank.csv', header = None, names = columns)
```
```
bank.info()
```
```
bank.describe()
```
# Naming all Numerical variables for easy analysis
```
num_cols = bank.select_dtypes(include=np.number).columns.tolist()
print("Numerical Variables:")
print(num_cols)
```
# VISUALIZING THE DISTRIBUTION OF EACH FEATURES AND THE CLASS BALANCE
### checking for skewness and outliers in our numeric variable 
```
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
```

![bank variance1](https://github.com/user-attachments/assets/742fb838-1a8b-4bef-8920-1b3c9031b2da)
![bank curtosis 2](https://github.com/user-attachments/assets/693ec3b6-8064-4823-91f4-cdd8ba633c6c)
![bank target 3](https://github.com/user-attachments/assets/53e15452-4609-44b3-9839-993960345e72)

# DATA PREPROCESSING
#checking for null value
```
bank.isnull().sum()
```
```
y = bank["Target"]
x = bank.drop (["Target"], axis = 1)
```
# SPLITING OF DATA TO TEST AND TRAIN 70/30
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

x_train.shape

x_test.shape

y_test.shape

y_train.shape
```
# SCALE THE FEATURE VALUE
```
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train = std.fit_transform(x_train)
X_test = std.transform(x_test)

```
# MODEL DEVELOPMENT
## LOGISTIC REGRESSION
```
from sklearn.linear_model import LogisticRegression
```
```
LGR = LogisticRegression(C=2,
                        n_jobs=5,
                        random_state=2)
```
```
LGR.fit(x_train, y_train)
```
```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the validation set
y_pred_lr = LGR.predict(x_test)
```

# Evaluate the model
```
print("Logistic Regression Evaluation:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
```
```
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_lr)
print("Accuracy:", accuracy)
```
```
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
y_pred_lr = LGR.predict(x_test)
cm = confusion_matrix(y_test, y_pred_lr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LGR.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Bank dataset")
plt.show()
```

![BANK CONFUSION LOG](https://github.com/user-attachments/assets/ed937f00-e5d0-4c2d-ae59-fc13b390a145)

# RANDOM FOREST
```
from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier?
```
```
rf = RandomForestClassifier(n_estimators=20)
```
```
rf.fit(x_train, y_train)
```
```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Predict on the validation set
y_pred = rf.predict(x_test)

# Evaluate the model
print("Logistic Regression Evaluation:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```
# Evaluate the model
```
y_pred = rf.predict(x_test)
accuracy_score(y_test, y_pred)
```
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#from sklearn.metrics import classification_report

#assuming 'knn' is your trained model, 'X_test' are your test features

y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for BANK dataset")
plt.show()
```
```
![bank Random conf](https://github.com/user-attachments/assets/ba49ab71-52dd-4044-a959-cbd03c79f92e)

# MODEL EVALUATION
## (i) Performance of models using a confusion matrix.
A confusion matrix is a table used to describe the performance of a classification model on a set of test data for which the true values are known. 
It helps in understanding the types of errors the model is making. Here's the typical arrangement of the confusion matrix along with the definitions 
of True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN):

## 1 Logistic regression
True Positive (TP): 226

True Negative(TN): 2

False Positive(FP): 183

False Negative(Fn): 3

## 2 Random forest
True Positive (TP): 229

True Negative(TN): 2

False Positive(FP): 183

False Negative(Fn): 3

True Positives (TP) Algorithm correctly predicted authentic banknote
True Negatives (TN) Algorithm correctly predicted counterfeit bank note,
False Positives (FP) Algorithm predicted authentic bank note, but its counterfeit
False Negatives (FN) Algorithm predicted counterfeit bank note, but its Authentic
## Conclusion
Based on the provided TP values, the random forest seems to have a slight edge.

## (ii) STRENGTH AND WEAKNESS OF MODEL
### 1 Logistic Regression
Strengths:
Interpretable coefficients.

Computationally efficient.

Provides probabilistic outputs.

Weaknesses:
Assumes linear relationships.

Sensitive to outliers.

### 2 Random Forest
Strengths:
Captures non-linear relationships.

High performance and robustness.

Weaknesses:
Less interpretable.

More computationally intensive

## Performance Metrics
Logistic Regression: Have lower accuracy and same F1-score on complex, non-linear data, but offers clear interpretation and efficiency.

Random Forest: have higher accuracy, same precision and recall, and ROC-AUC on complex datasets but at the cost of interpretability and computational resources.

# MODEL OPTIMIZATION USING THE GRIDSEARCH
```
from sklearn.model_selection import train_test_split, GridSearchCV
model = RandomForestClassifier(random_state=42)
```
```
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
```
```
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
```
```
best_model = grid_search.best_estimator_
```
```
y_pred = best_model.predict(x_test)
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
# CONCLUSION
### (i) FINDINGS IN THE PERFORMANCCE OF THE RANDOM FOREST USING THE GRID SEARCH
The Random Forest classifier outperformed Logistic Regression across all performance metrics, including accuracy (0.99), precision (1), recall (0.99), and F1 score (1). These results indicate that Random Forest is more effective in classifying banknotes as authentic or counterfeit.

### (ii) Insights into which model is more suitable for this task and why.
Using grid search to optimize the Random Forest model significantly enhances its performance by fine-tuning hyperparameters like the number of trees and maximum depth. This optimization leads to better handling of complex data patterns and interactions compared to Logistic Regression. The Random Forest's ensemble nature and robustness to overfitting make it more suitable for accurately classifying banknotes as authentic or counterfeit. Its superior performance metrics, including accuracy, precision, recall, and F1 score, indicate its effectiveness and reliability for this task. Thus, the optimized Random Forest model is the more suitable choice.

### (iii) Discuss any potential improvements or future work that could be done.
Using grid search techniques with the Random Forest model can significantly enhance performance by fine-tuning hyperparameters such as the number of trees, maximum depth, and minimum samples per leaf. Implementing feature engineering strategies, like creating new features and selecting the most relevant ones, can further boost model accuracy. Ensemble methods, such as stacking with other algorithms like XGBoost, can improve predictive power. Regularly retraining the model with new data ensures it remains effective over time. Monitoring and evaluating the model in real-world scenarios will help maintain its performance. Finally, incorporating additional relevant data sources and using interpretability tools like SHAP values can provide deeper insights and improve model reliability.

# FEATURE IMPORTANCE
```
#bank.Series(rf.feature_importance_, index=x.columns).nlargest(10)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
```
# TRAIN A RANDOM FOREST CLASSIFIER
```
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x, y)
```
# PLOT THE FEATURE IMPORTANCE
```
plt.figure()
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), [x.columns[i] for i in indices], rotation=90)
plt.show()
```
# CROSS VALIDATION¶
