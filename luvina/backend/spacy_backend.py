import numpy as np
import spacy
from .common import remove_repeated_elements


spacy_dictionary = spacy.load('en_core_web_sm')


# parser = English()
#
#
# def get_pos_tag(sentence):
#    """ add part of speech tags for each string in sentence
#    args:
#        takes input as sentence
#    returns:
#        list containing strings with part of speech tags
#    """
#    parseText = parser(sentence)
#    pos_tag_list = []
#    for word in parseText:
#        pos_tag_list.append(word)
#        pos_tag_list.append(word.pos_)
#    return pos_tag_list
#
#
# def find_lemma(sentence):
#    """ add lemma for each string in sentence
#    args:
#        takes input as sentence
#    returns:
#        list containing strings with lemma of each string
#    """
#    parseText = parser(sentence)
#    lemma_list = []
#    for word in parseText:
#        lemma_list.append(word)
#        lemma_list.append(word.lemma_)
#    return lemma_list
#
#
# def get_lowercase_words(token):
#    """ convert token into lowercase
#    args:
#        takes input as tokens
#    returns:
#        token in lowercase
#    """
#    return token.lower_
#

def get_vector(word):
    """ get glove vector word representation
    args:
        word: string
    returns:
        300 dimensional numpy array
    """
    return spacy_dictionary(word).vector


def get_vectors(tokens, mode='glove'):
    """ get glove vector representations of all tokens
    args:
        tokens: list of strings
    returns:
        list of of 300 dimensional numpy arrays
    """
    if mode == 'glove':
        return [get_vector(token) for token in tokens]


def calculate_norm(vector):
    """ calculate norm of a vector
    args:
        vector: single dimensional numpy array
    returns:
        float norm value
    """
    return np.linalg.norm(vector)


def calculate_cosine_similarity(vector_1, vector_2):
    """ calculate cosine similarity between two vectors
    args:
        vector_1: numpy vector
        vector_2: numpy vector
    returns:
        float indicating cosine similarity
    """
    norm_1 = calculate_norm(vector_1)
    norm_2 = calculate_norm(vector_2)
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    return np.dot(vector_1, vector_2) / (norm_1 * norm_2)


def get_related_words(token, max_num=3):
    """ use glove embeddings to get close embedded words
    args:
        token: string
        max_num: maximum number of words to be returned
    returns: list of strings
    """
    spacy_token = spacy_dictionary(token)
    queries = [vocab for vocab in spacy_token.vocab if vocab.prob >= -15]
    queries = sorted(queries, key=lambda w: spacy_token.similarity(w),
                     reverse=True)
    queries = [query.lower_ for query in queries if query.lower_ != token]
    queries = remove_repeated_elements(queries)
    if len(queries) > max_num:
        queries = queries[:max_num]
    return queries
