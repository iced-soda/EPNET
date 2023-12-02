# Read the text file containing the matrix
with open(r'C:\Users\sally\Documents\Uni\Research Thesis A\Enh_vs_Genes_Dataset\Enh_vs_Genes_matrix_number.txt', 'r') as file:
    matrix_content = file.readlines()

# Extract a sample from testing dataset

selected_rows = matrix_content[0] + matrix_content[8]

# Remove the trailing newline character from the last line
if selected_rows.endswith('\n'):
    selected_rows = selected_rows.rstrip('\n')

# Save the first row to a new file
with open('sample_input.txt', 'w') as new_file:
    new_file.writelines(selected_rows)
