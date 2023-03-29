import os
import sys
sys.path.append("XBNet")

from . import db
from flask import Blueprint, current_app, render_template, request, redirect, url_for
from flask_login import login_required
import joblib
import pandas as pd
import torch
import torch.nn.functional as F
from XBNet.training_utils import predict, predict_proba


views = Blueprint('views', __name__)


# @views.route('/selection', methods=['GET', 'POST'])
# def selection():
#     return render_template("selection.html")


# @views.route('/index.html', methods=['GET', 'POST'])
# @login_required
# def index():
#     X = pd.DataFrame()
#     if request.method == "POST":
#         get_model = joblib.load("app/xbnet_models/model.pkl")
#         time = request.form.get("time")
#         v1 = request.form.get("v1")
#         v2 = request.form.get("v2")
#         v3 = request.form.get("v3")
#         v4 = request.form.get("v4")
#         v5 = request.form.get("v5")
#         amount = request.form.get("amount")
#         X = pd.DataFrame([[time, v1, v2, v3, v4, v5, amount]], columns = ["Time", "V1", "V2", "V3", "V4", "V5", "Amount"]).astype(float)
#         X.to_csv('user_data.csv', index=None)
#         prediction = predict(get_model,X.to_numpy()[0,:])
#         return redirect(url_for('prediction_output', pred=prediction))
#     else:
#         prediction = ""
#     return render_template("index.html")

# Flask server-side code

@views.route('/upload_page.html', methods=['GET', 'POST'])
@login_required
def upload_page():

    if request.method == 'POST':

        # Get the file from the form request
        file = request.files['file']

        # If a file is selected
        if file:
            filename = file.filename
            # Save the file to the server
            file.save(os.path.join(current_app.instance_path, "user_data.csv"))

            # # Load the file using pandas
            # X = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))

            # # Unpickle the classifier
            # get_model = joblib.load(os.path.join(current_app.instance_path, "v2.pkl"))

            # # Get the prediction
            # prediction = predict(get_model, X.to_numpy()[0,:])
            # print(prediction)

            # Redirect the user to the prediction_output page
            return redirect(url_for('views.prediction_output'))

    return render_template('upload_page.html')


@views.route('/prediction_output/table')
@login_required
def prediction_output():
    # Load the file using pandas
    X = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))

    # Unpickle the classifier
    get_model = joblib.load(os.path.join(current_app.instance_path, "v2.pkl"))

    # Get the prediction
    prediction = predict(get_model, X.to_numpy()[0,:])
    print(prediction)

    proba = predict_proba(get_model, X.to_numpy()[0,:])
    print(proba)

    proba_tensor = torch.tensor([proba], dtype=torch.float)
    probs = F.softmax(proba_tensor, dim=0)

    positive_prob = 1 - probs.item()
    negative_prob = probs.item()

    print("Probability of Negative class (Non-Fraudulent Activity):", negative_prob)
    print("Probability of Positive class (Fraudulent Activity):", positive_prob)

    user_data = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))
    if prediction == 1:
        nfa = 'Non-Fraudulent Activity'
        message = f"The Model is {negative_prob:.2%} sure that it is {nfa}!"
        if user_data.empty:
            return render_template('prediction_output.html', tables=[], titles=[''], output=nfa, model_says = message)
        else:
            return render_template('prediction_output.html', tables=[user_data.to_html()], titles=[''], output=nfa, model_says = message)
    else:
        fa = 'Fraudulent Activity'
        message = f"The Model is {positive_prob:.2%} sure that it is {fa}!"
        if user_data.empty:
            return render_template("prediction_output.html", tables=[], titles=[''], output=fa, model_says = message)
        else:
            return render_template("prediction_output.html", tables=[user_data.to_html()], titles=[''], output=fa, model_says = message)

@views.route('/access_models.html')
@login_required
def access_models():

    return render_template('access_models.html')

@views.route('/data_analysis.html')
@login_required
def data_analysis():

    return render_template('/data_analysis.html')

@views.route('/my_documents.html')
@login_required
def my_documents():
    all_file_names = os.listdir(current_app.instance_path)
    csv_file_names = [f for f in all_file_names if f.endswith('.csv')]
    most_recent_user_data = csv_file_names[0]

    if request.method == 'POST':

        # Get a new file from the form request
        file = request.files['file']

        if file:

            # Save the file to the server
            file.save(os.path.join(current_app.instance_path, "user_data_new.csv"))

            return redirect(url_for('views.prediction_output'))

    return render_template('my_documents.html', recent_file=most_recent_user_data)