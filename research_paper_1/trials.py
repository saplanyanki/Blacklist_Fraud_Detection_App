import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from training_utils import training,predict
from models import XBNETClassifier
from run import run_XBNET
import joblib

data = pd.read_csv('test/creditcard.csv')
#this is the actual one
#data = data.iloc[0:4000, :]
#This is done for flask app trials
data = data.iloc[0:4000, [0,1,2,3,4,5,29,30]]
#print(data)
print(data.shape)
x_data = data[data.columns[:-1]]
print(x_data.shape)
y_data = data[data.columns[-1]]
le = LabelEncoder()
y_data = np.array(le.fit_transform(y_data))
print(le.classes_)

X_train,X_test,y_train,y_test = train_test_split(x_data.to_numpy(),y_data,test_size = 0.3,random_state = 0)
model = XBNETClassifier(X_train,y_train,2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

m,acc, lo, val_ac, val_lo = run_XBNET(X_train,X_test,y_train,y_test,model,criterion,optimizer,30,10)
#last two parameters are batch size and epoch
joblib.dump(model, "research_paper_1/xbnet_models/model.pkl")
#save the model

print(predict(m,x_data.to_numpy()[0,:]))