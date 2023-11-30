import pandas as pd 
from os.path import join

from config_path import INTERACTIONS_PATH

interactions_path = INTERACTIONS_PATH

def true_enhancers(file_name, threshold):

    file_name = join(interactions_path, file_name)
    data = pd.read_csv(file_name, sep='\t')

    data['Pvalue'] = pd.to_numeric(data['Pvalue'], errors='coerce')

    results_df = pd.DataFrame({
        'Enh':data['Enh'],
        'response':data.groupby('Enh')['Pvalue'].transform('max') >= threshold
    })

    results_df['response'] = results_df['response'].replace(True, 1)
    results_df['response'] = results_df['response'].replace(False, 0)

    results_df = results_df.drop_duplicates(subset='Enh')

    return results_df

