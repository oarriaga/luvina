from __future__ import absolute_import
from collections import Counter
from collections import OrderedDict
from itertools import chain
import numpy as np
from .enchant_backend import in_dictionary
from .enchant_backend import suggest_words
from .nltk_backend import make_ngrams
from .nltk_backend import calculate_levenshtein_distance


def correct_misspelling(token, distance_threshold=2):
    """ correct misspelling of token by comparing suggestions and measuring
    the levenshtein distance
    args:
        token: string
        distance_threshold: int that describes the maximum number of desired
        changed characters from 'token'
    returns:
        best_suggestion: string containing suggestion
        token: original token is not match was found
    """
    if in_dictionary(token):
        return token
    suggested_words = suggest_words(token)
    if suggested_words is not None:
        num_modified_characters = []
        for suggested_word in suggested_words:
            distance = calculate_levenshtein_distance(token, suggested_word)
            num_modified_characters.append(distance)
        # this min is showing errors since it takes an empy/none variable as inputen    
        min_num_modified_characters = min(num_modified_characters)
        best_arg = num_modified_characters.index(min_num_modified_characters)
        if distance_threshold > min_num_modified_characters:
            best_suggestion = suggested_words[best_arg]
            return best_suggestion
        else:
            return token
    else:
        return token


def remove_repeated_elements(tokens):
    """ removes repeated tokens
    args:
        tokens: a list of tokens
    returns:
        filtered_tokens: a list of tokens without repeated tokens
    """
    filtered_tokens = list(OrderedDict((token, None) for token in tokens))
    return filtered_tokens


def join(tokens):
    """ construct a string sentence from joining tokens with spaces
    args:
        tokens: list of strings
    returns:
        joined_tokens: a string containing all tokens
    """
    joined_tokens = ' '.join(tokens)
    return joined_tokens


def compute_jaccard_similarity(query, document):
    """ function taken explicitly from:
    http://billchambers.me/tutorials/2014/12/21/tf-idf-explained-in-python.html
    calculates the intersection over union of a query
    in a given document.
    """
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def calculate_jaccard_coefficient(a, b):
    union = list(set(a + b))
    intersection = list(set(a) - (set(a) - set(b)))
    jaccard_coefficient = float(len(intersection)) / len(union)
    return jaccard_coefficient


def correct_misspelling_ngram(token, levenshtein_treshold=3):
    """ corrects token by suggesting words and filtering them
    using the levenhstein distance. Then it takes all filtered
    words and chooses the one with the highest jaccard coefficient
    calculated using bigrams.
    args:
        token: string
        levenshtein threshold: int
    returns:
        token: string
    """
    if in_dictionary(token):
        return token
    suggested_words = suggest_words(token)
    jaccard_coefficients = []
    best_suggested_words = []
    if suggested_words is not None:
        token_bigrams = make_ngrams(token, 2)
        for suggested_word in suggested_words:
            distance = calculate_levenshtein_distance(token, suggested_word)
            if distance < levenshtein_treshold:
                suggested_bigrams = make_ngrams(suggested_word, 2)
                jaccard_coefficient = calculate_jaccard_coefficient(
                                    token_bigrams, suggested_bigrams)
                jaccard_coefficients.append(jaccard_coefficient)
                best_suggested_words.append(suggested_word)
        highest_jaccard = max(jaccard_coefficients)
        best_arg = jaccard_coefficients.index(highest_jaccard)
        word = best_suggested_words[best_arg]
        return word
    else:
        return word


def get_token_frequencies(sentences):
    """ count the number of times all tokens appear in all sentences.
    args:
        sentences: list of sentences where each sentence contains a
        list of tokens
    returns:
        word_frequencies: list containing two elements lists with the
        word (string) and an integer describing the amount of times
        the word has appeared in the all the sentences.
    """
    word_frequencies = Counter(chain(*sentences)).most_common()
    return word_frequencies


def pad(tokens, max_token_size=22, remove=False, BOS_token='<BOS>',
        EOS_token='<EOS>', PAD_token='<PAD>'):
    """
    args:
        tokens: a list of strings containing tokens
        max_token_size: Max token size including the EOS and BOS tokens.
        remove: Boolean flag for determining if it should remove tokens
        bigger than max_token_size.
        BOS_token: string beginning of the sentence token.
        EOS_token: string end of the sentence token.
        PAD_token: string pad token.
    returns:
        tuple of strings containing all added/removed tokens.
        Tokens will get removed if the remove flag is enabled
        and the number of tokens is bigger than max_token_size.
    """

    sentence = list(tokens)
    max_sentence_length = max_token_size - 2
    if len(sentence) == max_sentence_length:
        padded_sentence = [BOS_token] + sentence + [EOS_token]
    elif len(sentence) > max_sentence_length and not remove:
        sentence = sentence[:max_sentence_length]
        padded_sentence = [BOS_token] + sentence + [EOS_token]
    elif len(sentence) > max_sentence_length and remove:
        return []
    elif len(sentence) < max_sentence_length:
        padded_sentence = [BOS_token] + sentence + [EOS_token]
        pad_size = max_token_size - len(padded_sentence)
        pad = [PAD_token] * pad_size
        padded_sentence = padded_sentence + pad
    return padded_sentence


def remove_infrequent_tokens(sentences, word_frequencies, min_frequency=3):
    pass
#######################################################
# Deprecated functions
#######################################################

# def remove_long_sentences(sentences, max_token_size=25):
#    # TODO: Add y predictions that should/could also be filtered
#    # you pass y and iterate over its first dimension.
#    """ removes sentences with a length bigger that max_length.
#    args:
#        sentences: list of lists containing strings/tokens.
#        max_length: int > 0
#    returns:
#        filtered_sentences: list pf lists containing a strings/tokens.
#    """
#    filtered_sentences = []
#    for tokens in sentences:
#        if len(tokens) <= max_token_size:
#            filtered_sentences.append(tokens)
#    return filtered_sentences
#
#
# def remove_long_sentences(sentences, associated_data=None,
#    max_token_size=25):
#    # TODO: Add y predictions that should/could also be filtered
#    # you pass y and iterate over its first dimension.
#    """ removes sentences with a length bigger that max_length.
#    args:
#        sentences: list of lists containing strings/tokens.
#        data: Additional associated data that should be removed
#        if a sentence is removed e. g. labels or another pair of sentences
#        max_length: int > 0
#    returns:
#        filtered_sentences: list pf lists containing a strings/tokens.
#    """
#    mask = get_token_size_mask(sentences, max_token_size)
#    sentences = np.asarray(sentences)[mask]
#    associated_data = np.asarray(associated_data)[mask]

def get_token_size_mask(sentences, max_token_size=25):
    """ returns mask containing True for sentences with less
    or equal amount of tokens than the max_token_size, and
    false otherwise.
    args:
        sentences: list of lists containing strings/tokens
        max_length: int > 0
    returns:
        boolean numpy array of size(sentences)
    """
    mask = np.zeros(shape=len(sentences))
    for sentence_arg, tokens in enumerate(sentences):
        if len(tokens) <= max_token_size:
            mask[sentence_arg] = True
        else:
            mask[sentence_arg] = False
    return mask


def mask_data(data, mask):
    if data is not np.ndarray:
        data = np.asarray(data)
        return data[mask].tolist()
    else:
        return data[mask]


def pad_with_zeros(sentences, max_length=25):
    data = []
    for vectors in sentences:
        vectors = np.asarray(vectors)
        sentence_length, embedding_dimension = vectors.shape
        missing_zeros = max_length - sentence_length
        zero_array = np.zeros(shape=(missing_zeros, embedding_dimension))
        vectors = np.concatenate((vectors, zero_array), axis=0)
        data.append(vectors.tolist())
    return data
