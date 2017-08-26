from .enchant_backend import in_dictionary
from .enchant_backend import suggest_words
from .nltk_backend import get_distance


def correct_misspelling(token, distance_threshold=2):
    """ correct misspelling of token by comparing suggestions and measuring
    the amount of different characters
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
            distance = get_distance(token, suggested_word)
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
