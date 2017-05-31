from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
from collections import OrderedDict
from itertools import chain

"""
nltk.download(punkt)
nltk.download(wordnet)
nltk.download(stopwords)
"""

#FIXME: load a stem class
#stemmer = PorterStemmer()

def tokenize(sentence):
    #FIXME: this tokenizer removes repeated words.
    return word_tokenize(sentence)

def get_synonyms(word):
    #assert we are getting a string
    synonyms = []
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def filter_stop_words(tokenized_sentence):
    #tokenized_sentence = tokenize_sentence(sentence)
    english_stops = set(stopwords.words('english'))
    filtered_tokens = []
    for token in tokenized_sentence:
        if token not in english_stops:
            filtered_tokens.append(token)
    return filtered_tokens

"""
def stem_words(tokenized_sentence):
    stemmed_sentence = []
    for token in tokenized_sentence:
        stemmed_sentence.append(stemmer.stem(token))
    return stemmed_sentence
"""

def filter_repeated_words(tokenized_sentence):
    # FIXME this probably destroys the order
    return list(set(tokenized_sentence))

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def calculate_wordnet_similarity(word_1, word_2):
    word_synset_1 = wordnet.synset(word_1)
    word_synset_2 = wordnet.synset(word_1)
    return word_synset_1.wup_similiraty(word_synset_2)

def find_synonyms_set(tokenized_sentence_1, tokenized_sentence_2):
    synonyms = []
    for token in tokenized_sentence_1:
        token_synonyms = set(get_synonyms(token))
        found_synonyms = token_synonyms.intersection(set(tokenized_sentence_2))
        synonyms.append(list(found_synonyms))

def find_synonyms(reference_sentence, hypothesis_sentence, return_tokens=True):
    """
    Inputs: two lists of tokenized string sentences
    Returns: A dictionary with keys being the word in the reference_sentence
    and the values a list of the arguments where there is a synonym in the
    hypothesis_sentence.
    BUGS:
    - What to do with repeated words, which are repeated keys and get
    re-written?
    - Indices get repeated use:
        sentence_1 = 'there is a cat and the cat sat on a mat'
        sentence_2 = 'there is a mat and on the mat sat the cat'
    - The words 'and' ant 'the' in the previous examples without tokenization
    don't get connected.
    """
    synonyms_connections = OrderedDict()
    """FIXME: it will search even when the words are repeated it will
    overwrite the key value using the same word token as key.
    """
    for reference_token in reference_sentence:
        reference_synonyms = list(set(get_synonyms(reference_token)))
        reference_synonyms = reference_synonyms + [reference_token]
        hypothesis_args = []
        for hypothesis_arg, hypothesis_token in enumerate(hypothesis_sentence):
            """FIXME: the problem happens when the words are identical it finds
            that all the synonyms are synonyms with the word.
            UPDATE: Solved but untested
            """
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

def edit(hypothesis_sentence, token_connections):
    edited_hypothesis = []
    hypothesis_highlights = list(chain(*list(token_connections.values())))
    for hypothesis_token in hypothesis_sentence:
        if hypothesis_token in hypothesis_highlights:
            hypothesis_token = '**' + hypothesis_token
        edited_hypothesis.append(hypothesis_token)
    edited_hypothesis = ' '.join(edited_hypothesis)
    return edited_hypothesis


