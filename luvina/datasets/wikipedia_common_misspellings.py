import numpy as np
from luvina.utils.data_utils import get_file


file_name = "wikipedia_common_misspellings.txt"
origin = ('https://raw.githubusercontent.com/rameshjesswani/Semantic-Textual' +
          '-Similarity/master/nlp_basics/nltk/dataset/dataset_misspelled.txt')


def load_data():
    """
    i. Data set is downloaded from:
    https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/A
    ii. Data set contains misspelled words and correct words
    iii. Words are selected randomly from data set
    iv. Around 670 words are selected randomly from data set to evaluate
    the performance of spell correctors
    """
    file_path = get_file(file_name, origin)
    load_file = np.loadtxt(file_path, dtype='str')

    return load_file
