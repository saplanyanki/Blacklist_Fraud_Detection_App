import joblib
import pandas as pd
from training_utils import training,predict

get_model = joblib.load("research_paper_1/xbnet_models/model.pkl")
print(get_model)
X = pd.DataFrame([[1, 1, -1, 0, 0, -1, 100]], columns = ["Time", "V1", "V2", "V3", "V4", "V5", "Amount"]).astype(float)
print(X)
prediction = predict(get_model,X.to_numpy()[0,:])
print(prediction)
print (X.dtypes)