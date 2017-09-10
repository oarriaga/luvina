import luvina.backend as luv


def correct(token):
    token = luv.remove_repeated_characters(token)
    token = luv.correct_misspelling(token)
    return token


sentence = 'woooow languge ist veri complicaated'
print('Original incorrect sentence: \n', sentence)
corrected_sentence = [correct(token) for token in luv.tokenize(sentence)]
print('Corrected sentence: \n', luv.join(corrected_sentence))
