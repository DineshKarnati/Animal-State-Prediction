# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:02:05 2020

@author: hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing


df = pd.read_csv(r"C:\Users\hp\Desktop\ZS\train.csv")
df['sex_upon_outcome'] = df['sex_upon_outcome'].fillna('Unknown')
df['sex_upon_intake'] = df['sex_upon_intake'].fillna('Unknown')
id = df['animal_id_outcome']
df =df.drop(['age_upon_intake','count','age_upon_intake_(years)','age_upon_intake_age_group','intake_datetime','intake_monthyear',
             'time_in_shelter','age_upon_outcome','date_of_birth','age_upon_outcome_(years)','age_upon_outcome_age_group','outcome_datetime',
             'outcome_monthyear'],axis = 1)
df.columns
df = df.drop(['animal_id_outcome', 'breed'],axis = 1)

color = df['color'].str.get_dummies(sep='/').rename(lambda x: 'col_' + x, axis='columns')
df = pd.concat([df,color],axis = 1)
####################converting categorical variables to numerical#########################
intake_condition = pd.get_dummies(df['intake_condition'],drop_first = True)
intake_type = pd.get_dummies(df['intake_type'],drop_first = True)
animal_type = pd.get_dummies(df['animal_type'],drop_first = True)
sex_upon_intake = pd.get_dummies(df['sex_upon_intake'],drop_first = True).rename(lambda x: 'in_' + x, axis='columns')
sex_upon_outcome = pd.get_dummies(df['sex_upon_outcome'],drop_first = True).rename(lambda x: 'out_' + x, axis='columns')
df = pd.concat([df,sex_upon_outcome,sex_upon_intake,animal_type,intake_type,intake_condition],axis=1)
df =df.drop(['sex_upon_outcome','color','sex_upon_intake', 'animal_type', 'intake_type', 'intake_condition' ],axis=1)
#########################
df =df.drop(['col_Orange Tiger', 'col_Gray Tiger', 'col_Ruddy',],axis= 1)
intake_weekday = pd.get_dummies(df['intake_weekday'],drop_first = True)
outcome_weekday = pd.get_dummies(df['outcome_weekday'],drop_first = True)
y = df['outcome_type']

df = df.drop(['intake_weekday', 'outcome_type','outcome_weekday'],axis = 1)
 
X = df.values
num_of_classes = len(y.unique())
print(num_of_classes)
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

xgb = XGBClassifier(booster='dart', objective='multi:softprob', max_depth = 8, random_state=42, eval_metric="merror", num_class=num_of_classes)
# Fit the classifier with the training data
xgb.fit(X_train,y_train)
val = xgb.predict(X_test)
#########################################
lb = preprocessing.LabelBinarizer()
lb.fit(y_test)
##########################################
y_test_lb = lb.transform(y_test)
val_lb = lb.transform(val)
##########################################
roc_auc_score(y_test_lb, val_lb, average='macro')
########################################
output = pd.DataFrame()
output['Expected Output'] = y_test
output['Predicted Output'] = val
output.head()
#####################################
from sklearn.metrics import f1_score
f1_score(y_test_lb, val_lb, average='micro')


############################################################################


test = pd.read_csv(r"C:\Users\hp\Desktop\ZS\test.csv")

animal_id_outcome = test['animal_id_outcome']

color = test['color'].str.get_dummies(sep='/').rename(lambda x: 'col_' + x, axis='columns')
test = pd.concat([test,color],axis = 1)
test =test.drop(['age_upon_intake','count','breed', 'animal_id_outcome', 'age_upon_intake_(years)','age_upon_intake_age_group','intake_datetime','intake_monthyear',
             'time_in_shelter','age_upon_outcome','date_of_birth','age_upon_outcome_(years)','age_upon_outcome_age_group','outcome_datetime',
             'outcome_monthyear'],axis = 1)

intake_condition = pd.get_dummies(test['intake_condition'],drop_first = True)
intake_type = pd.get_dummies(test['intake_type'],drop_first = True)
animal_type = pd.get_dummies(test['animal_type'],drop_first = True)
sex_upon_intake = pd.get_dummies(test['sex_upon_intake'],drop_first = True).rename(lambda x: 'in_' + x, axis='columns')
sex_upon_outcome = pd.get_dummies(test['sex_upon_outcome'],drop_first = True).rename(lambda x: 'out_' + x, axis='columns')
test = pd.concat([test,sex_upon_outcome,sex_upon_intake,animal_type,intake_type,intake_condition],axis=1)
test =test.drop(['sex_upon_outcome','color','sex_upon_intake', 'animal_type', 'intake_type', 'intake_condition' ],axis=1)
#########################
intake_weekday = pd.get_dummies(test['intake_weekday'],drop_first = True)
outcome_weekday = pd.get_dummies(test['outcome_weekday'],drop_first = True)
test = test.drop(['intake_weekday','outcome_weekday'],axis = 1)

testing = test.values
yyy = xgb.predict(testing)
output = pd.DataFrame({'animal_id_outcome': animal_id_outcome, 'outcome_type': yyy})
output.to_csv(r'C:\Users\hp\Desktop\ZS\XGBoost_jdart+.csv', index=False)
