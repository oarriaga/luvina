import luvina.backend as luv


def correct(token):
    corrected_token = luv.remove_repeated_characters(token)
    corrected_token = luv.correct_misspelling(corrected_token)
    return corrected_token


sentence = 'woooow languge ist veri complicaated'
print(sentence)
tokens = luv.tokenize(sentence)
corrected_sentence = [correct(token) for token in tokens]
print(corrected_sentence)
