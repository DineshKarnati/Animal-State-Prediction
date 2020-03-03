# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:21:38 2020

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

#EXPLORING THE DATA

#############Importing the data########################################
df = pd.read_csv(r"C:\Users\hp\Desktop\ZS\train.csv")

############EXPLORING THE DATA############################
df['sex_upon_outcome'] = df['sex_upon_outcome'].fillna('Unknown')# NULL VALUES
df['sex_upon_intake'] = df['sex_upon_intake'].fillna('Unknown')   #NULL VALUES

###############VISUALIZATION OF TARGET AND DEPOENDENCIES#####
AnimalTypeGroup = df.groupby('animal_type')
print(AnimalTypeGroup.animal_type.value_counts())
AnimalTypeGroup.outcome_type.value_counts()

AnimalTypeGraph = df[['animal_type','outcome_type']].groupby(['outcome_type','animal_type']).size().unstack()
AnimalTypeGraph.plot(kind='bar',color=['r','c','g','b'])
##############################################################3

id = df['animal_id_outcome']   #ID

############################DROPPING COLUMNS########################

df =df.drop(['age_upon_intake','count','age_upon_intake_(years)','age_upon_intake_age_group','intake_datetime','intake_monthyear',
             'time_in_shelter','age_upon_outcome','date_of_birth','age_upon_outcome_(years)','age_upon_outcome_age_group','outcome_datetime',
             'outcome_monthyear'],axis = 1)
df.columns
df = df.drop(['animal_id_outcome', 'breed'],axis = 1)

###########################CREATING DUMMY COLOR COLUMNS#################

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
####################################################
df =df.drop(['col_Orange Tiger', 'col_Gray Tiger', 'col_Ruddy',],axis= 1)
intake_weekday = pd.get_dummies(df['intake_weekday'],drop_first = True)
outcome_weekday = pd.get_dummies(df['outcome_weekday'],drop_first = True)

#############################TARGET VARIABLE####################

y = df['outcome_type']   ####TARGET VARIABLE

df = df.drop(['intake_weekday', 'outcome_type','outcome_weekday'],axis = 1)
 
X = df.values
num_of_classes = len(y.unique())
print(num_of_classes)

# ###################  Split into training and test sets  ##############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#####################  XGB CLASSIFIER MODEL  #########################

xgb = XGBClassifier(max_depth=8, learning_rate=0.02, n_estimators=500, objective='multi:softprob',
                        subsample=0.8, colsample_bytree=0.8, nthread=1)# Fit the classifier with the training data

###################T   RAINING THE MODEL  #######################
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
#################  DATAFRAME OF EXPECTED AND PREDICTED  #######################
output = pd.DataFrame()
output['Expected Output'] = y_test
output['Predicted Output'] = val
output.head()
####################  F1SCORE  #################
from sklearn.metrics import f1_score
f1_score(y_test_lb, val_lb, average='micro')


##############################  TEST SET     ##############################################

""" SAME STEPS FOLLOWS IN TEST SET FROM TRAIN DATASET"""

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
###############   PREDICTING RESULTS   ####################3
yyy = xgb.predict(testing)
output = pd.DataFrame({'animal_id_outcome': animal_id_outcome, 'outcome_type': yyy})
output.to_csv(r'C:\Users\hp\Desktop\ZS\XGBoost_FINAL.csv', index=False)
