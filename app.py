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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(project_root, 'backend_database.sql')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

filename = './_logs/enh_vs_genes/log/fs/P-net.h5'

params_file = './train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing.py'

# load db
print("linking to database")
db = SQLAlchemy(app)

# load model 
print("1")
loader = importlib.machinery.SourceFileLoader('params', params_file)
params = loader.load_module()
model_params_ = deepcopy(params.models[0])
model = nn.Model(**model_params_['params'])
model.load_model(filename)

# Set the upload folder
# UPLOAD_FOLDER = '~/9450/9450_MainProjectWeb/uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}

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

# If without CORS (and wepbage files located within EPNET model directory)
# @app.route('/')
# def index():
#     return render_template('run_model.html')

### Login Routes ###
@app.route('/register', methods=['POST'])
def register():

    data = request.get_json()
    email = data['email']
    password = data['password']
    first_name = data['firstname']
    last_name = data['lastname']
    gender = data['gender']
    dob = data['dob']
    institute = data['institute']
    specialties = data['specialties']
    user = User(email=email, first_name=first_name, last_name=last_name, gender=gender, dob=dob, institute=institute, specialties=specialties)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Registration successful", "redirect": url_for('reg_success')})

@app.route('/login', methods=['GET', 'POST'])
def login():
    print("Going here to login")
    data = request.get_json()
    if request.method == 'POST':
        email = data['email']
        password = data['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            print("the password is correct")
            return jsonify({"message": "Login successful", "redirect": url_for('dashboard')})
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/reg_success')
def reg_success():
#   email = request.args.get('email')
#   first_name = request.args.get('first_name')
# , email=email, first_name=first_name
    return render_template('regSuccess.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    clinicians = Clinician.query.all()
    clinicians_data = [{'id': clinician.id, 'name': clinician.name} for clinician in clinicians]
    return render_template('dashboard.html', clinicians=clinicians_data)

###---###

@app.route('/index')
def home():
    return render_template('index_html')

@app.route('/run_model')
def run_model():
    return render_template('run_model.html')

@app.route('/documentation')
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
