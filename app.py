from flask import Flask, request, jsonify, make_response, render_template
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

app = Flask(__name__)
CORS(app)

# Add the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

filename = 'EPNET/_logs/enh_vs_genes/log/fs/P-net.h5'

params_file = 'EPNET/train/params/P1000/pnet/onsplit_average_reg_10_tanh_large_testing.py'

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

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# If without CORS (and wepbage files located within EPNET model directory)
# @app.route('/')
# def index():
#     return render_template('run_model.html')

@app.route('/index')
def home():
    return render_template('index_html')

@app.route('/run_model')
def run_model():
    return render_template('run_model.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/regSuccess')
def regSuccess():
    return render_template('regSucess.html')

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
    matrix_data = np.array(data)[:, 1:]

    # Convert the data to a numeric type if needed
    matrix_data = matrix_data.astype(float)
    
    return matrix_data

if __name__ == '__main__':
    app.run(debug=True)
