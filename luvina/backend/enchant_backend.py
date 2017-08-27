import enchant

# -------------------------------------------------------------------
# ENCHANT GLOBAL VARIABLES
# -------------------------------------------------------------------
spell_dictionary = enchant.Dict('en')
# -------------------------------------------------------------------


def in_dictionary(token):
    """ checks if a token is found in the dictionary or not
    args:
        token: string
    returns:
        boolean determining if the token was found in dictionary
    """
    return spell_dictionary.check(token)


def suggest_words(token):
    """ suggest similar tokens
    args:
        token: string
    returns:
        list of strings containing similar tokens
    """
    return spell_dictionary.suggest(token)
