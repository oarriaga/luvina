import luvina.backend as luv

sentence = 'the languge is very complicated'
tokens = luv.tokenize(sentence)
corrected_sentence= [luv.correct_misspelling(token) for token in tokens]
print(corrected_sentence)
