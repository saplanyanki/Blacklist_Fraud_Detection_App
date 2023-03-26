import sys
sys.path.append("XBNet")
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from training_utils import training,predict
from models_xb import XBNETClassifier
from run import run_XBNET
from torch.optim.lr_scheduler import StepLR
import joblib
from modifications import *
from flask import current_app

#data = pd.read_csv('test/creditcard.csv')
data = df_smote
#this is the actual one
#data = data.iloc[0:100000, :]
#This is done for flask app trials
#data = data.iloc[0:4000, [0,1,2,3,4,5,29,30]]
#print(data)
print(data.shape)
x_data = data[data.columns[:-1]]
print(x_data.shape)
y_data = data[data.columns[-1]]
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))
print(le.classes_)

X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.3,random_state = 12)

# get the value counts of each class in y_train
train_counts = np.unique(y_train, return_counts=True)
print("Train class counts:", dict(zip(le.inverse_transform(train_counts[0]), train_counts[1])))

# get the value counts of each class in y_test
test_counts = np.unique(y_test, return_counts=True)
print("Test class counts:", dict(zip(le.inverse_transform(test_counts[0]), test_counts[1])))
model = XBNETClassifier(X_train,y_train,4)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,32,20)
#last two parameters are batch size and epoch

#save the model
joblib.dump(model, os.path.join(current_app.instance_path, "model.pkl"))
print(predict(m,x_data.to_numpy()[0,:]))