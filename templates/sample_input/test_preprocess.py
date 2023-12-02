import numpy as np

# Step 1: Read the file and extract the data
file_path = 'sample_input.txt'

with open(file_path, 'r') as file:
    # Skip the first line
    header_line = file.readline()
    lines = file.readlines()

# Step 2: Tokenize the data
data = [line.split('\t') for line in lines]

# Step 3: Convert the data into a NumPy array
# Assuming the first element of each line is not part of the matrix
matrix_data = np.array(data)[:, 1:]

# Convert the data to a numeric type if needed
matrix_data_numeric = matrix_data.astype(float)

print(matrix_data_numeric)
