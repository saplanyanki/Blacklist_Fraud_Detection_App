import sys
sys.path.append("XBNet")
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from training_utils import training,predict
from models import XBNETClassifier
from run import run_XBNET
import joblib
from imblearn.over_sampling import SMOTE


train_d = pd.read_csv('test/fraudTrain.csv')
test_d = pd.read_csv('test/fraudTest.csv')

df = pd.concat([test_d.assign(ind="test"), train_d.assign(ind="train")])
df = df.dropna()

df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop('cc_num', axis=1, inplace=True)
df.drop('first', axis=1, inplace=True)
df.drop('last', axis=1, inplace=True)
df.drop('job', axis=1, inplace=True)
df.drop('dob', axis=1, inplace=True)
df.drop('trans_num', axis=1, inplace=True)
df.drop('merchant', axis=1, inplace=True)
df.drop('ind', axis=1, inplace=True)
df.drop('unix_time', axis=1, inplace=True)
df.drop('merch_lat', axis=1, inplace=True)
df.drop('merch_long', axis=1, inplace=True)
df.drop('street', axis=1, inplace=True)
df.drop('lat', axis=1, inplace=True)
df.drop('long', axis=1, inplace=True)
df.drop('trans_date_trans_time', axis=1, inplace=True)
df.drop('city_pop', axis=1, inplace=True)
df.drop('city', axis=1, inplace=True)
df.drop('state', axis=1, inplace=True)

def genders(row):
  if row['gender'] == 'M':
        val = 0
  else:
      val = 1
  return val

df['genders'] = df.apply(genders, axis=1)
df.drop('gender', axis=1, inplace=True)

def create_group_columns(data):
    groups = {
        "Essentials": ["gas_transport", "grocery_pos", "home", "personal_care"],
        "Leisure": ["entertainment", "shopping_pos", "shopping_net"],
        "Wellness": ["health_fitness", "food_dining"],
        "Other": ["kids_pets", "misc_pos", "misc_net", "grocery_net"]
    }

    result = pd.DataFrame()

    for group, categories in groups.items():
        result[group] = data['category'].apply(lambda x: 1 if x in categories else 0)

    df = pd.concat([data, result], axis=1)
    return df
df = create_group_columns(df)
df.drop('category', axis=1, inplace=True)

# load data into a Pandas DataFrame

# split the data into features and target variable
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# create a SMOTE object and fit it to the data
smote = SMOTE(sampling_strategy=0.2)
X_smote, y_smote = smote.fit_resample(X, y)

# create a new DataFrame with the resampled data
df_smote = pd.concat([X_smote, y_smote], axis=1)

cols = [col for col in df_smote.columns if col != 'is_fraud'] + ['is_fraud']
df_smote = df_smote[cols]