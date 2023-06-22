import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# import csv file fetal_health.csv 
df = pd.read_csv('fetal_health.csv')

# Preprocessing data:
# 1. Check for missing values
# print(df.isna().sum().sort_values())

# 2. Seperate features and target:
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# 3. Split data into train and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12,stratify=y)

# 4. Scale data:
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)

# 5. Evaluate model:

kf=KFold(n_splits=5,shuffle=True,random_state=42)
models={
    'knn':KNeighborsClassifier(),
    'svc':SVC(),
    'dtc':DecisionTreeClassifier(),
    'rfc':RandomForestClassifier(),
    'gbc':GradientBoostingClassifier()
}
results=[]

# for model in models.values():
#     result=cross_val_score(model,x_train_scaled,y_train,cv=kf)
#     results.append(result)

# plt.boxplot(results,labels=models.keys())
# plt.show()

# ✅ from the cross validation results, we can see that the best model is GradientBoostingClassifier


