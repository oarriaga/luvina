import luvina.backend as luv

sentence = 'Alway rememer decmber and oovember'
print(sentence)
tokens = luv.tokenize(sentence)
corrected_sentence= [luv.spell_corrector_ngram(token) for token in tokens]
print(corrected_sentence)
