from flask import Flask, request, jsonify, make_response, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
import os
import sys
import numpy as np
import pandas as pd
import importlib
from copy import deepcopy
from model import nn
from flask import Flask
from flask_cors import CORS
import json

# login and password 
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///backend_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# load db
print("linking to database")
db = SQLAlchemy(app)
# CORS(app)

# Setting a secret key for assigning session cookie
secret_key = os.urandom(16)
print(secret_key)
app.secret_key = 'your_random_secret_key_here'

filename = './_logs/enh_vs_genes/log/fs/P-net.h5'

params_file = './train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing.py'

###---###
# load model 
loader = importlib.machinery.SourceFileLoader('params', params_file)
params = loader.load_module()
model_params_ = deepcopy(params.models[0])
model = nn.Model(**model_params_['params'])
model.load_model(filename)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.String(100), nullable=False)
    institute = db.Column(db.String(200), nullable=True)
    specialties = db.Column(db.String(200), nullable=True)
    password = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)
    
    def __repr__(self):
        return f'<User {self.first_name} ID {self.id}>'

class Clinician(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(100), nullable=False)
    job_title = db.Column(db.String(100), nullable=False)
    institute = db.Column(db.String(200), nullable=True)
    specialties = db.Column(db.String(200), nullable=True)

with app.app_context():
    db.create_all()

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

### Login Routes ###
@app.route('/register', methods=['POST'])
def register():
    print('attempted to register')
    email = request.form.get('email')
    password = request.form.get('password')
    first_name = request.form.get('firstname')
    last_name = request.form.get('lastname')
    gender = request.form.get('gender')
    dob = request.form.get('dob')
    institute = request.form.get('institute')
    specialties = request.form.get('specialties')
    user = User(email=email, first_name=first_name, last_name=last_name, gender=gender, dob=dob, institute=institute, specialties=specialties)
    user.set_password(password)

    # Check for existing email address: 
    if User.query.filter_by(email=email).first() is not None:
        return jsonify({"message": "The Email address has been registered"}),401
    
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Registration successful", "redirect": url_for('reg_success')})

@app.route('/login', methods=['POST'])
def login():
    print('attempted to login')
    email = request.form.get('email-input')
    password = request.form.get('password-input')
    user = User.query.filter_by(email=email).first()
    print(user)
    if user and user.check_password(password): # validated password
        session['user_id'] = user.id
        print("the password is correct")
        return jsonify({"message": "Login successful", "redirect": url_for('dashboard')})
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/regSuccess.html')
def reg_success():
#   email = request.args.get('email')
#   first_name = request.args.get('first_name')
# , email=email, first_name=first_name
    return render_template('regSuccess.html')

@app.route('/dashboard.html')
def dashboard():
    clinicians = Clinician.query.all()
    return render_template('dashboard.html', clinicians=clinicians)

@app.route('/logout')
def logout():
    # Remove user_id from session
    session.pop('user_id', None)
    # Redirect to login page, home page, or any other page
    return redirect(url_for('home'))  # Replace 'login' with the endpoint you want to redirect to

ALLOWED_EXTENSIONS = {'csv', 'txt'}

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/run_model.html')
def run_model():
    return render_template('run_model.html')

@app.route('/documentation.html')
def documentation():
    return render_template('documentation.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['fileInput']
    print("1")
    # Check if the file is provided and has an allowed extension
    if file and allowed_file(file.filename):

        print(file)

        # Perform preprocessing on the file
        processed_data = process_uploaded_file(file)

        # Make predictions using the loaded model
        prediction = model.predict(processed_data)

        # Convert the numpy array to a list (or nested list) before returning
        prediction_list = prediction.tolist()

        # Return the result as JSON
        return jsonify({'prediction': prediction_list})
    else:
        return jsonify({'error': 'Invalid file or file format'})

def process_uploaded_file(file):

    # Step 1: Read the file and extract the data
    # Skip the first line
    header_line = file.readline()
    lines = [line.decode('utf-8') for line in file.readlines()]

    # Step 2: Tokenize the data
    data = [line.split('\t') for line in lines]

    # Step 3: Convert the data into a NumPy array
    # Assuming the first element of each line is not part of the matrix
    # Getting the first 5 rows to preview 
    matrix_data = np.array(data)
    num_rows, num_cols = matrix_data.shape
    if num_rows > 5: 
        matrix_data = np.array(data)[:, 1:6]
    else:
         # Preview all data s
         matrix_data = np.array(data)[:, 1:]

    # Convert the data to a numeric type if needed
    matrix_data = matrix_data.astype(float)
    
    return matrix_data

if __name__ == '__main__':
    app.run(debug=True)
