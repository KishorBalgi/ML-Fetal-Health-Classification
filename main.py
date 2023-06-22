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

# 6. Hyperparameter tuning:
# 6.1. GridSearchCV:
param_grid={
    'n_estimators':[100,200,300,400,500],
    'learning_rate':[0.01,0.1,1,10,100]
}
gbc=GradientBoostingClassifier()
grid=GridSearchCV(gbc,param_grid=param_grid,cv=kf)
grid.fit(x_train_scaled,y_train)
print(grid.best_params_)
print(grid.best_score_)

# ✅ from the GridSearchCV results, we can see that the best parameters are n_estimators=400 and learning_rate=0.1

# 6.2. RandomizedSearchCV:
# gridRCV=RandomizedSearchCV(gbc,param_distributions=param_grid,cv=kf,n_iter=2)
# gridRCV.fit(x_train_scaled,y_train)
# print(gridRCV.best_params_)
# print(gridRCV.best_score_)

# ✅ from the RandomizedSearchCV results, we can observe that since the number of iterations is small, the best parameters are not the same as the GridSearchCV results. However, the best score is the same as the GridSearchCV results.


