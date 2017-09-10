import luvina.backend as luv


def preprocess_token(token):
    token = luv.expand_contractions(token)
    token = luv.correct_misspelling_ngram(token)
    token = luv.get_vector(token)
    return token


sentence = "I can't remember when was the last time I took the dog outside"
tokens = luv.tokenize(sentence)
vectors = [preprocess_token(token) for token in tokens]
