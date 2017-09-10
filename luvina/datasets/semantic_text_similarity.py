import pandas as pd
from luvina.utils.data_utils import get_file


origin = 'https://raw.githubusercontent.com/alvations/stasis/master/sts.csv'
file_name = 'semantic_text_similarity.csv'


def load_data():
    filepath = get_file(file_name, origin)
    data = pd.read_csv(filepath, sep='\t', error_bad_lines=False,
                       warn_bad_lines=False)
    data = data.dropna()
    return data
