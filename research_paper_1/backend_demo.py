from flask import Flask, request, render_template, url_for, redirect, request
from training_utils import training,predict
import pandas as pd
import joblib

# Declare a Flask app
app = Flask(__name__)

# Main function here
@app.route('/', methods=['GET', 'POST'])

def main():
    
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
        
        # Get prediction
        prediction = predict(get_model,X.to_numpy()[0,:])
        return redirect(url_for('prediction_output', pred=prediction))

    else:
        prediction = ""
    
    return render_template("index.html")
    

# Add a new route for the prediction_output page
@app.route('/prediction_output/<int:pred>')
def prediction_output(pred):
    if pred == 1:
        nfa = 'Non-Fraudalent Activity'
        return render_template('prediction_output.html', output = nfa)
    else:
        fa = 'Fraudalent Activity'
        return render_template("prediction_output.html", output = fa)


# Running the app
if __name__ == '__main__':
    app.run(debug = True)
    #server listens local host on http://127.0.0.1:5000/