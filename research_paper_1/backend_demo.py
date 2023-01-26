from flask import Flask, request, render_template, url_for, redirect, request, session
from training_utils import training,predict
import pandas as pd
import joblib

# Declare a Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Main function here
@app.route('/', methods=['GET', 'POST'])

def main():
    X = pd.DataFrame()
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        get_model = joblib.load("research_paper_1/xbnet_models/model.pkl")
        
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
        X.to_csv('user_data.csv', index=None)
        # Get prediction
        prediction = predict(get_model,X.to_numpy()[0,:])

        return redirect(url_for('prediction_output', pred=prediction))

    else:
        prediction = ""
    return render_template("index.html")
    



# Add a new route for the prediction_output page
@app.route('/prediction_output/<int:pred>/table')
def prediction_output(pred):
    user_data = pd.read_csv('user_data.csv')
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
# def prediction_output(pred):
#     user_data = pd.read_csv('user_data.csv')
#     if pred == 1:
#         nfa = 'Non-Fraudalent Activity'
#         return render_template('prediction_output.html', tables=[user_data.to_html()], titles=[''], output = nfa)
#     else:
#         fa = 'Fraudalent Activity'
#         return render_template("prediction_output.html", tables=[user_data.to_html()], titles=[''],  output = fa)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)
    #server listens local host on http://127.0.0.1:5000/