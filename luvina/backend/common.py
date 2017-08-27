from __future__ import absolute_import
from collections import OrderedDict
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
