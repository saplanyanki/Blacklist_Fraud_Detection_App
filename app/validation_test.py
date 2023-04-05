import sys
sys.path.append("XBNet")

from training_utils import validate
from run import Data
import pandas as pd
import numpy
import joblib
import torch
from torch.utils.data import Dataset, DataLoader


validation_df = pd.read_csv("val4.csv")
v4 = joblib.load("instance/v4.pkl")


val_x = validation_df.drop('is_fraud', axis=1)
val_y = validation_df['is_fraud']


val_x = val_x.to_numpy()
val_y = val_y.to_numpy()


validation_df = DataLoader(Data(val_x, val_y), batch_size=32)


criterion = torch.nn.BCEWithLogitsLoss()
validate(v4, validation_df, criterion, 5, True)