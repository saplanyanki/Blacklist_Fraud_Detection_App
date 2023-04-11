import sys
sys.path.append("XBNet")
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from training_utils import training,predict
from models_xb import XBNETClassifier
from run import run_XBNET
import joblib
from imblearn.over_sampling import SMOTE

train_d = pd.read_csv('fraudTrain.csv')
test_d = pd.read_csv('fraudTest.csv')

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
# df.drop('lat', axis=1, inplace=True)
# df.drop('long', axis=1, inplace=True)
df.drop('trans_date_trans_time', axis=1, inplace=True)
df.drop('city_pop', axis=1, inplace=True)
df.drop('city', axis=1, inplace=True)
# df.drop('state', axis=1, inplace=True)

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

one_hot = pd.get_dummies(df['state'])
#one_hot_m = pd.get_dummies(df['merchant'])
df = df.join(one_hot)
#df = df.join(one_hot_m)
df = df.drop('state', axis=1)
#df.drop('zip', axis=1, inplace=True)
print(len(df.index))

# remove val data / sets of 4 - 5k
val1 = df.iloc[0:5000, :]
val1.to_csv('val1.csv', index=False)
val2 = df.iloc[5000:10000:, :]
val2.to_csv('val2.csv', index=False)
val3 = df.iloc[10000:15000:, :]
val3.to_csv('val3.csv', index=False)
val4 = df.iloc[15000:20000:, :]
val4.to_csv('val4.csv', index=False)
val_large = df.iloc[80000:2800000, :]
val_large.to_csv('val_large.csv', index=False)

# load data into a Pandas DataFrame
df = df.iloc[20000:80000, :]
# split the data into features and target variable
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# create a SMOTE object and fit it to the data
smote = SMOTE(sampling_strategy=1)
X_smote, y_smote = smote.fit_resample(X, y)

# create a new DataFrame with the resampled data
df_smote = pd.concat([X_smote, y_smote], axis=1)

cols = [col for col in df_smote.columns if col != 'is_fraud'] + ['is_fraud']
df_smote = df_smote[cols]
df_smote = df_smote.sample(frac=1, random_state=4)
df_smote['amt'] = df_smote['amt'].round(1)
print(df_smote.is_fraud.value_counts())

df_smote.to_csv('data.csv', index=False) # index=False to exclude the index column
