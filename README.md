# Bank
Develop a machine learning model to accurately classify whether a banknote is authentic, or counterfeit based on various extracted features
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
