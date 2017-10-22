import re
from collections import OrderedDict

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk import pos_tag
from nltk.metrics import edit_distance
from nltk.util import ngrams

# -------------------------------------------------------------------
# NLTK GLOBAL VARIABLES
# -------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would')
]

patterns = []
for regular_expression, replacement in replacement_patterns:
    patterns.append((re.compile(regular_expression), replacement))

repeat_regular_expresssion = re.compile(r'(\w*)(\w)\2(\w*)')
repeat_replacement = r'\1\2\3'
# -------------------------------------------------------------------


def download_nltk_data(package_name='all'):
    """ download necessary data from NLTK
    args:
        package_name: string containing the package name to install
    returns:
        None
    """
    if package_name is 'all':
        data = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        for package in data:
            download(package)
    else:
        download(package)


def tokenize(sentence, lowercase=True):
    """ tokenize a setence
    args:
        sentence: string
        lowercase: boolean flag for returning lowercase format
    returns:
        a list of the tokens in the sentence
    """
    tokens = word_tokenize(sentence)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def _remove_repeated_elements(tokens):
    """ removes repeated tokens
    args:
        tokens: a list of tokens
    returns:
        filtered_tokens: a list of tokens without repeated tokens
    """
    filtered_tokens = list(OrderedDict((token, None) for token in tokens))
    return filtered_tokens


def get_synonyms(token):
    """ get synonyms of word using wordnet
    args:
        token: string
    returns:
        synonyms: list containing synonyms as strings
    """
    synonyms = []
    if len(wordnet.synsets(token)) == 0:
        return None
    for synset in wordnet.synsets(token):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    synonyms = _remove_repeated_elements(synonyms)
    synonyms.remove(token)
    return synonyms


def get_definition(token):
    """ get definition of word using wordnet
    args:
        token: string
    returns:
        definition: string containing definition of word
    """
    synsets = wordnet.synsets(token)
    if len(synsets) == 0:
        return []
    return synsets[0].definition()


def get_stop_words(language='english'):
    """ get English stop words
    args:
        language: language for which the stop words will be used
    returns:
        stop_words= list of stop words in string format.
    """
    stop_words = list(set(stopwords.words(language)))
    return stop_words


def filter_tokens(tokens, filters):
    """ eliminates tokens contained in filters
    args:
        tokens: list of tokens to be filtered.
        filters: list of tokens that will get removed from 'tokens'.
    returns:
        filtered_tokens: a list of tokens that don't contain any
        token from filters.
    """
    filtered_tokens = []
    for token in tokens:
        if token not in filters:
            filtered_tokens.append(token)
    return filtered_tokens


def calculate_wordnet_similarity(token_1, token_2):
    """ calculates similarity between two tokens using wordnet
    args:
        token_1: string
        token_2: string
    returns:
        wordnet_similarity: int score for wordnet similarity or None
        if either given token is not a word.
    """
    synset_1 = wordnet.synsets(token_1)
    synset_2 = wordnet.synsets(token_2)
    if len(synset_1) == 0 or len(synset_2) == 0:
        return None
    wordnet_similarity = synset_1[0].wup_similarity(synset_2[0])
    return wordnet_similarity


def lemmatize(token, pos='n'):
    """ lemmatize a token using a selected part of speech
    args:
        token: string
        pos = parth_of_speech of the given token
    returns
        lemmatized_token: a lemmatized/reduced token
    """
    lemmatized_token = lemmatizer.lemmatize(token, pos=pos)
    return lemmatized_token


def expand_contractions(sentence):
    """ expand English contractions found in sentence/token:
    args:
        sentence: string of complete sentence or a single token
    returns:
        sentence = string sentence/token without contractions
    """
    for (pattern, replacement) in patterns:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence


def remove_repeated_characters(token):
    """ removes recursively repeated characters in a token if token not
    found in dictionary
    args:
        token: string
    returns:
        replaced_token: string
    """
    if wordnet.synsets(token):
        return token
    replaced_token = repeat_regular_expresssion.sub(repeat_replacement, token)
    if replaced_token != token:
        return remove_repeated_characters(replaced_token)
    else:
        return replaced_token


def tag_pos(tokens):
    """ add parts of speech tags for every token given
    args:
        tokens: list of strings
    returns
        list of lists containing string token and string part of speech
    """
    return pos_tag(tokens)


def calculate_levenshtein_distance(token_1, token_2):
    """ get number of characters that need to be substituted,
    inserted or deleted to go from token_1 to token_2.
    args:
        token_1: string
        token_2: string
    returns:
        int
    """
    return edit_distance(token_1, token_2)


def make_ngrams(sequence, n, pad_left=False, pad_right=False,
                left_pad_symbol=None, right_pad_symbol=None):
    """ return the ngrams made by sequence of items
    args:
        tokens: list of strings/tokens
        n: number of subsequent items
    returns:
        list of lists
    """
    return list(ngrams(sequence, n, pad_left, pad_right,
                       left_pad_symbol, right_pad_symbol))
