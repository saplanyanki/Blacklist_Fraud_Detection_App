# import sys
# sys.path.append("XBNet")

# from flask import Flask, request, render_template, url_for, redirect
# from flask_bcrypt import Bcrypt
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
# from flask_wtf import FlaskForm
# from wtforms import StringField, PasswordField, SubmitField
# from wtforms.validators import InputRequired, Length, ValidationError
# from XBNet.training_utils import predict, training 
# import pandas as pd
# import joblib


# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
# app.config['SECRET_KEY'] = 'secret_key'
# db = SQLAlchemy(app)
# app.app_context().push()
# db.create_all()


# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'


# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))


# How to update tables:
# (In terminal)
# Change working directory to app
# Use the following commands in terminal:
# >>> from app import app, db
# >>> app.app_context().push()
# >>> db.create_all()
# >>> exit()


# Add text to register page that informs the user
# that they can only have x chars in their user/pass
# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key = True)
#     username = db.Column(db.String(20), nullable = False, unique = True)
#     password = db.Column(db.String(80), nullable = False)


# class LoginForm(FlaskForm):
#     username = StringField(validators = [InputRequired(), Length(min = 4, max = 20)], render_kw={"placeholder": "Username"})
#     password = PasswordField(validators = [InputRequired(), Length(min = 8, max = 20)], render_kw={"placeholder": "Password"})
#     submit = SubmitField('Login')


# class RegisterForm(FlaskForm):
#     username = StringField(validators = [InputRequired(), Length(min = 4, max = 20)], render_kw = {"placeholder": "Username"})
#     password = PasswordField(validators = [InputRequired(), Length(min = 8, max = 20)], render_kw = {"placeholder": "Password"})
#     submit = SubmitField('Register')

#     def validate_username(self, username):
#             existing_user_username = User.query.filter_by(username=username.data).first()
#             if existing_user_username:
#                 raise ValidationError('That username already exists.')


# @app.route('/', methods=['GET', 'POST'])
# def login():
#     form = LoginForm()
#     if form.validate_on_submit():
#         user = User.query.filter_by(username=form.username.data).first()
#         if user:
#             if Bcrypt.check_password_hash(user.password, form.password.data):
#                 login_user(user)
#                 return redirect(url_for('selection'))
#     return render_template('login.html', form = form)


# @app.route('/register')
# def register():
#     form = RegisterForm()

#     if form.validate_on_submit():
#         hashed_password = Bcrypt.generate_password_hash(form.password.data)
#         new_user = User(username = form.username.data, password = hashed_password)
#         db.session.add(new_user)
#         db.session.commit()
#         return redirect(url_for('login'))

#     return render_template('register.html', form = form)


# @app.route('/selection')
# @login_required
# def selection():
#     return render_template('selection.html')

# @app.route('/index.html', methods=['GET', 'POST'])
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
    
# @app.route('/index2.html', methods=['GET', 'POST'])
# @login_required
# def index2():
#     if request.method == 'POST':
#         # Get the file from the form request
#         file = request.files['file']

#         # If a file is selected
#         if file:
#             # Save the file to the server
#             file.save(file.filename)

#             # Load the file using pandas
#             X = pd.read_csv(file.filename)

#             # Unpickle the classifier
#             get_model = joblib.load("app/xbnet_models/model.pkl")

#             # Get the prediction
#             prediction = predict(get_model, X.to_numpy()[0,:])
#             print(prediction)
#             # Save the user's data to a file
#             X.to_csv('app/user_file/user_data.csv', index=None)
#             # Redirect the user to the prediction_output page
#             return redirect(url_for('prediction_output', pred=prediction))

#     return render_template('index2.html')

# This is bugged
# Add a new route for the prediction_output page
# @app.route('/prediction_output/<int:pred>/table')
# @login_required
# def prediction_output(pred):
#     user_data = pd.read_csv('app/user_file/user_data.csv')
#     if pred == 1:
#         nfa = 'Non-Fraudalent Activity'
#         if user_data.empty:
#             return render_template('prediction_output.html', tables=[], titles=[''], output = nfa)
#         else:
#             return render_template('prediction_output.html', tables=[user_data.to_html()], titles=[''], output = nfa)
#     else:
#         fa = 'Fraudalent Activity'
#         if user_data.empty:
#             return render_template("prediction_output.html", tables=[], titles=[''],  output = fa)
#         else:
#             return render_template("prediction_output.html", tables=[user_data.to_html()], titles=[''],  output = fa)


# @app.route('/logout', methods=['GET', 'POST'])
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))


# Running the app
# if __name__ == '__main__':
#     app.run(debug = True)
#server listens local host on http://127.0.0.1:5000/