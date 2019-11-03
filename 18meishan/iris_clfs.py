# coding:utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
import os
from time import time

data = pd.read_csv('iris.data', header=None)
x = data[[0,1,2,3]]
y = LabelEncoder().fit_transform(data[4])
print(data[4])
np.set_printoptions(linewidth=52)
print(y)
print()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

if False:# os.path.exists('iris.model'):
    print('加载模型...')
    model = joblib.load('iris.model')
else:
    print('训练支持向量机模型...')
    # model = SVC(kernel='rbf')
    model = LogisticRegression(penalty='l2', C=1)
    # model_cv = GridSearchCV(model, cv=3, param_grid={
    #     'C': np.logspace(-4, 4, 5),
    #     'gamma': np.logspace(-4, 4, 5)
    # })
    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()
    model = DecisionTreeClassifier( )
    # model = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=3)
    # model.fit(x_train, y_train)
    print('耗时：', t_end - t_start)
    joblib.dump(model, 'iris.model')

y_train_pred = model.predict(x_train)
print('训练集正确率：', accuracy_score(y_train, y_train_pred))
y_test_pred = model.predict(x_test)
print('测试集正确率：', accuracy_score(y_test, y_test_pred))