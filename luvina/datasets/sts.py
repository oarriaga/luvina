import pandas as pd


def load_data(filepath='semantic_text_similarity.csv'):
    # dataset_path = 'datasets/'
    # filename = dataset_path + data_file
    data = pd.read_csv(filepath, sep='\t', error_bad_lines=False,
                       warn_bad_lines=False)
    # data['Score'] = data['Score'].fillna(0.0)
    # the dataset might be bias towards 0.0 if I fill them without knowing
    data = data.dropna()
    return data
