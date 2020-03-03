# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:46:12 2020

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
df.shape
df.isnull().any()
df['sex_upon_outcome'] = df['sex_upon_outcome'].fillna('Unknown')
df['sex_upon_intake'] = df['sex_upon_intake'].fillna('Unknown')
df.columns
df['outcome_type'].unique()
df['outcome_type'].value_counts()

AnimalTypeGroup = df.groupby('animal_type')
print(AnimalTypeGroup.animal_type.value_counts())
AnimalTypeGroup.outcome_type.value_counts()

AnimalTypeGraph = df[['animal_type','outcome_type']].groupby(['outcome_type','animal_type']).size().unstack()
AnimalTypeGraph.plot(kind='bar',color=['r','c','g','b'])