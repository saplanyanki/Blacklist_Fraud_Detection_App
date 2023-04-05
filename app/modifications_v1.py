####Use for Old & Original RP data
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


cred_card = pd.read_csv('test/creditcard.csv')
cred_card = cred_card.dropna()

cred_card.drop('Time', axis=1, inplace=True)

# load data into a Pandas DataFrame
cred_card = cred_card.iloc[0:40000, :]
# split the data into features and target variable
X = cred_card.drop('Class', axis=1)
y = cred_card['Class']

# create a SMOTE object and fit it to the data
smote = SMOTE(sampling_strategy=1)
X_smote, y_smote = smote.fit_resample(X, y)

# create a new DataFrame with the resampled data
cc_smote = pd.concat([X_smote, y_smote], axis=1)

cols = [col for col in cc_smote.columns if col != 'Class'] + ['Class']
cc_smote = cc_smote[cols]
cc_smote = cc_smote.sample(frac=1, random_state=17)
cc_smote['Amount'] = cc_smote['Amount'].round(1)
print(cc_smote.Class.value_counts())

cc_smote.to_csv('cc_v15.csv', index=False) # index=False to exclude the index column