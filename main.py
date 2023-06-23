import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix,classification_report

# import csv file fetal_health.csv 
df = pd.read_csv('./data/fetal_health.csv')
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
# param_grid={
#     'n_estimators':[100,200,300,400,500],
#     'learning_rate':[0.01,0.1,1,10,100]
# }
# gbc=GradientBoostingClassifier()
# grid=GridSearchCV(gbc,param_grid=param_grid,cv=kf)
# grid.fit(x_train_scaled,y_train)
# print("GridSearchCV results:")
# print("Best parameters: ")
# print(grid.best_params_)
# print("Best score: ")
# print(grid.best_score_)

# ✅ from the GridSearchCV results, we can see that the best parameters are n_estimators=400 and learning_rate=0.1

# 6.2. RandomizedSearchCV:
# gridRCV=RandomizedSearchCV(gbc,param_distributions=param_grid,cv=kf,n_iter=2)
# gridRCV.fit(x_train_scaled,y_train)
# print("RandomizedSearchCV results:")
# print("Best parameters: ")
# print(gridRCV.best_params_)
# print("Best score: ")
# print(gridRCV.best_score_)

# ✅ from the RandomizedSearchCV results, we can observe that since the number of iterations is small, the best parameters are not the same as the GridSearchCV results. However, the best score is the same as the GridSearchCV results.

# 7. Evaluate model with the best parameters:
gbc=GradientBoostingClassifier(n_estimators=400,learning_rate=0.1)
gbc.fit(x_train_scaled,y_train)
print("Accuracy score: ")
print(gbc.score(x_test_scaled,y_test))

# 8. Score, Confusion matrix and Classification report:
y_pred=gbc.predict(x_test_scaled)
print("MSE: ",mean_squared_error(y_test,y_pred))
print("Confusion matrix: ")
print(confusion_matrix(y_test,y_pred))
print("Classification report: ")
print(classification_report(y_test,y_pred))

# ✅ from the results, we can see that the accuracy score is 0.9507042253521126

# 9. Save model:
pickle.dump(gbc,open('fetal_health_model.pkl','wb'))
