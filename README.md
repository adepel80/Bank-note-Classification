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
# import matplotlib.pyplot as plt
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
