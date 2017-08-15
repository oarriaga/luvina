import numpy as np


def get_data(data_file='dataset_misspelled_words.txt'):
    '''
    i. Data set is downloaded from:
    https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/A
    ii. Data set contains misspelled words and correct words
    iii. Words are selected randomly from data set
    iv. Around 670 words are selected randomly from data set to evaluate the performance of spell correctors
    '''
    load_file = np.loadtxt(data_file,dtype='str')

    return load_file
