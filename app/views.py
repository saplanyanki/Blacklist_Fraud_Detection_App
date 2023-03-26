import os
import sys
sys.path.append("XBNet")

from flask import Blueprint, current_app, render_template, request, flash
from flask_login import login_required, current_user
from . import db


###
from flask import Flask, request, render_template, url_for, redirect
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from XBNet.training_utils import predict, training 
import pandas as pd
import joblib
###


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


@views.route('/upload_page.html', methods=['GET', 'POST'])
@login_required
def upload_page():

    if request.method == 'POST':
        # Get the file from the form request
        file = request.files['file']

        # If a file is selected
        if file:
            # Save the file to the server
            file.save(os.path.join(current_app.instance_path, "user_data.csv"))

            # Load the file using pandas
            X = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))

            # Unpickle the classifier
            get_model = joblib.load("app/xbnet_models/model.pkl")

            # Get the prediction
            prediction = predict(get_model, X.to_numpy()[0,:])
            print(prediction)

            # Save the user's data to a file
            X.to_csv(os.path.join(current_app.instance_path, "user_data_file.csv"))

            # Redirect the user to the prediction_output page
            return redirect(url_for('prediction_output', pred=prediction))

    return render_template('upload_page.html')


@views.route('/prediction_output/<int:pred>/table')
@login_required
def prediction_output(pred):
    user_data = pd.read_csv(os.path.join(current_app.instance_path, "user_data.csv"))
    if pred == 1:
        nfa = 'Non-Fraudalent Activity'
        if user_data.empty:
            return render_template('prediction_output.html', tables=[], titles=[''], output = nfa)
        else:
            return render_template('prediction_output.html', tables=[user_data.to_html()], titles=[''], output = nfa)
    else:
        fa = 'Fraudalent Activity'
        if user_data.empty:
            return render_template("prediction_output.html", tables=[], titles=[''],  output = fa)
        else:
            return render_template("prediction_output.html", tables=[user_data.to_html()], titles=[''],  output = fa)