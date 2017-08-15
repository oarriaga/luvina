from __future__ import division
import numpy as np
import luvina.backend as luv
import load_misspelled_dataset
from load_misspelled_dataset import *

def check_misspelledWords_minimumEditDistance(dataset):
    '''
    i. This function takes dataset downloaded from Wikipedia as input
    ii. Dataset contains misspelled words and correct words
    iii. Each misspelled word is corrected using minimum edit distance(MED) spell corrector
    iv. Each corrected word using MED spell corrector is compared with the correct word in dataset
    '''
    number_of_corrected_words = 0
    for i in range(len(dataset)):

        corrected_word = luv.correct_misspelling(dataset[i][0])
        # corrected_word = luv.correct_misspelling_ngram(dataset[i][0])
        # if suggested word by ngram spell corrector is equal to correct word in data set
        if corrected_word == dataset[i][2]:
            number_of_corrected_words += 1
    print "============================================================================================"
    print "Total number of misspelled words in database", len(dataset)
    print "Total number of corrected words", number_of_corrected_words
    print "Total percentage ", (number_of_corrected_words/len(dataset)) * 100
    print "============================================================================================"


def check_misspelledWords_ngram(dataset):
    '''
    i. This function takes dataset downloaded from Wikipedia as input
    ii. Dataset contains misspelled words and correct words
    iii. Each misspelled word is corrected using ngram spell corrector
    iv. Each corrected word using ngram spell corrector is compared with the correct word in dataset
    '''
    number_of_corrected_words = 0
    for i in range(len(dataset)):

        corrected_word = luv.correct_misspelling_ngram(dataset[i][0])
        # if suggested word by ngram spell corrector is equal to correct word in data set
        if corrected_word == dataset[i][2]:
            number_of_corrected_words += 1
    print "============================================================================================"
    print "Total number of misspelled words in database", len(dataset)
    print "Total number of corrected words", number_of_corrected_words
    print "Total percentage ", (number_of_corrected_words/len(dataset)) * 100
    print "============================================================================================"


if __name__== "__main__":
    load_dataset = get_data()
    check_misspelledWords_minimumEditDistance(load_dataset)
    # check_misspelledWords_ngram(load_dataset)
