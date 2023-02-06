import sys
sys.path.append('research_paper_1')

from flask import Flask, request, render_template, g, redirect, session, url_for
from training_utils import training,predict
import pandas as pd
import joblib

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

## This is temporary before we create a database of user/pass combinations
users = []
users.append(User(id=1, username='efreedman56', password='1234'))
users.append(User(id=2, username='Yanki', password='Saplan'))

# Declare a Flask app
app = Flask(__name__)
app.secret_key = 'lkajbvdlkjncq'

# @app.before_first_request
# def before_request():
#     g.user = None
#     if 'user_id' in session:
#         user = [x for x in users if x.id == session['user_id']][0]
#         g.user = user

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        # Removes session if you are logged into multiple tabs
        session.pop('user_id', None)

        username = request.form['username']
        password = request.form['password']
        
        # Checks to see if any username matches the input username
        # Will fail if the username does not exist in the database
        user = [x for x in users if x.username == username][0]

        # If user and pass are correct, redirect to fraud url
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('fraud'))

        # If login fails, go back to login url
        return redirect(url_for('login'))

    return render_template('login.html')

# Main function here
@app.route('/fraud', methods=['GET', 'POST'])
def fraud():
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        get_model = joblib.load("flask_app/xbnet_models/model.pkl")
        
        # Get values through input bars
        time = request.form.get("time")
        v1 = request.form.get("v1")
        v2 = request.form.get("v2")
        v3 = request.form.get("v3")
        v4 = request.form.get("v4")
        v5 = request.form.get("v5")
        amount = request.form.get("amount")

        # Put inputs to dataframe
        X = pd.DataFrame([[time, v1, v2, v3, v4, v5, amount]], columns = ["Time", "V1", "V2", "V3", "V4", "V5", "Amount"]).astype(float)
        
        # Get prediction
        prediction = predict(get_model,X.to_numpy()[0,:])

    else:
        prediction = ""
        
    #return render_template("index.html", output = prediction)
    if prediction == 1:
        nfa = 'Non-Fraudalent Activity'
        return render_template("index.html", output = nfa)
    else:
        fa = 'Fraudalent Activity'
        return render_template("index.html", output = fa)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
    #server listens local host on http://127.0.0.1:5000/
