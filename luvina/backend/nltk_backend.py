from collections import OrderedDict
from itertools import chain
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.metrics import edit_distance
import nltk
import enchant
#GLOBAL VARIABLES -----------------------------------
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

spell_dictionary = enchant.Dict('en')
# ----------------------------------------------------

def download_nltk_data(package_name=None):
    if package_name is None:
        data = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
        for package in data:
            download(package)
    else:
        download(package)

def tokenize(sentence):
    return word_tokenize(sentence)

def get_synonyms(word):
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def get_definition(word):
    synset = wordnet.synsets(word)[0]
    synset.definition()

def filter_stop_words(tokenized_sentence):
    english_stops = set(stopwords.words('english'))
    filtered_tokens = []
    for token in tokenized_sentence:
        if token not in english_stops:
            filtered_tokens.append(token)
    return filtered_tokens

def filter_repeated_words(tokenized_sentence):
    return list(set(tokenized_sentence))

def calculate_wordnet_similarity(word_1, word_2):
    word_synset_1 = wordnet.synset(word_1)
    word_synset_2 = wordnet.synset(word_1)
    return word_synset_1.wup_similiraty(word_synset_2)

def find_synonyms(reference_sentence, hypothesis_sentence, return_tokens=True):
    """
    Inputs: two lists of tokenized string sentences
    Returns: A dictionary with keys being the word in the reference_sentence
    and the values a list of the arguments where there is a synonym in the
    hypothesis_sentence.
    """
    synonyms_connections = OrderedDict()
    for reference_token in reference_sentence:
        reference_synonyms = list(set(get_synonyms(reference_token)))
        reference_synonyms = reference_synonyms + [reference_token]
        hypothesis_args = []
        for hypothesis_arg, hypothesis_token in enumerate(hypothesis_sentence):
            for reference_synonym in reference_synonyms:
                words_are_synonyms = hypothesis_token == reference_synonym
                words_are_the_same = hypothesis_token == reference_token
                if words_are_the_same:
                    if return_tokens:
                        hypothesis_args.append(hypothesis_token)
                    else:
                        hypothesis_args.append(hypothesis_arg)
                    break
                elif words_are_synonyms:
                    if return_tokens:
                        hypothesis_args.append(hypothesis_token)
                    else:
                        hypothesis_args.append(hypothesis_arg)

        synonyms_connections[reference_token] = hypothesis_args
    return synonyms_connections

def edit(hypothesis_sentence, token_connections,
            start_string='**', end_string='**'):
    edited_hypothesis = []
    hypothesis_highlights = list(chain(*list(token_connections.values())))
    for hypothesis_token in hypothesis_sentence:
        if hypothesis_token in hypothesis_highlights:
            hypothesis_token = start_string + hypothesis_token + end_string
        edited_hypothesis.append(hypothesis_token)
    edited_hypothesis = ' '.join(edited_hypothesis)
    return edited_hypothesis

def lemmatize(word, pos='n'):
    return lemmatizer.lemmatize(word, pos=pos)

def expand_contractions(sentence):
    for (pattern, replacement) in patterns:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence

def remove_repeated_characters(word):
    if wordnet.synsets(word):
        return word
    replaced_word = repeat_regular_expresssion.sub(repeat_replacement, word)
    if replaced_word != word:
        return  remove_repeated_characters(replaced_word)
    else:
        return replaced_word

def in_dictionary(word):
    return spell_dictionary.check(word)

def suggest_words(word):
    return spell_dictionary.suggest(word)

def correct_misspelling(word, max_distance=2):
    if in_dictionary(word):
        return word
    suggested_words = suggest_words(word)
    if suggested_words is not None:
        num_modified_characters = []
        for suggest_word in suggested_words:
            num_modified_characters.append(edit_distance(word, suggest_word))
        max_num_modified_characters = min(num_modified_characters)
        best_arg = num_modified_characters.index(max_num_modified_characters)
        if max_distance > max_num_modified_characters:
            best_suggestion = suggested_words[best_arg]
            return best_suggestion
        else:
            return word
    else:
        return word

def tag_pos(tokens):
    return nltk.pos_tag(tokens)

