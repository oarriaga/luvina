import luvina.backend as luv

sentence = 'languge is veri complicaated'
print(sentence)
tokens = luv.tokenize(sentence)
corrected_sentence= [luv.correct_misspelling(token) for token in tokens]
print(corrected_sentence)
