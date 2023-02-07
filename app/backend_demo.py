import sys
sys.path.append("XBNet")

from flask import Flask, request, render_template, url_for, redirect, request
from training_utils import training,predict
import pandas as pd
import joblib


####
# This is temporary before we create a database of user/pass combinations
class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

users = []
users.append(User(id=1, username='efreedman56', password='1234'))
users.append(User(id=2, username='Yanki', password='Saplan'))
####


# Declare a Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'


# This is the default page
# Need to add a redirect for the register button
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']
        
        # Check user/pass combination
        for u in users:
            if username == u.username and password == u.password:
                return redirect(url_for('selection'))

        # If login fails, go back to login url
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/selection')
def selection():
    return render_template('selection.html')

@app.route('/index.html', methods=['GET', 'POST'])
def index():
    X = pd.DataFrame()
    if request.method == "POST":
        get_model = joblib.load("app/xbnet_models/model.pkl")
        time = request.form.get("time")
        v1 = request.form.get("v1")
        v2 = request.form.get("v2")
        v3 = request.form.get("v3")
        v4 = request.form.get("v4")
        v5 = request.form.get("v5")
        amount = request.form.get("amount")
        X = pd.DataFrame([[time, v1, v2, v3, v4, v5, amount]], columns = ["Time", "V1", "V2", "V3", "V4", "V5", "Amount"]).astype(float)
        X.to_csv('user_data.csv', index=None)
        prediction = predict(get_model,X.to_numpy()[0,:])
        return redirect(url_for('prediction_output', pred=prediction))
    else:
        prediction = ""
    return render_template("index.html")
    
@app.route('/index2.html', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        # Get the file from the form request
        file = request.files['file']

        # If a file is selected
        if file:
            # Save the file to the server
            file.save(file.filename)

            # Load the file using pandas
            X = pd.read_csv(file.filename)

            # Unpickle the classifier
            get_model = joblib.load("app/xbnet_models/model.pkl")

            # Get the prediction
            prediction = predict(get_model, X.to_numpy()[0,:])
            print(prediction)
            # Save the user's data to a file
            X.to_csv('app/user_file/user_data.csv', index=None)
            # Redirect the user to the prediction_output page
            return redirect(url_for('prediction_output', pred=prediction))

    return render_template('index2.html')

# This is bugged
# Add a new route for the prediction_output page
@app.route('/prediction_output/<int:pred>/table')
def prediction_output(pred):
    user_data = pd.read_csv('app/user_file/user_data.csv')
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


# Running the app
if __name__ == '__main__':
    app.run(debug = True)
    #server listens local host on http://127.0.0.1:5000/