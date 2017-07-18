import luvina.backend as luv

sentence = 'Always rememer decmber and oovember'
print(sentence)
tokens = luv.tokenize(sentence)
corrected_sentence = [luv.correct_misspelling_ngram(token) for token in tokens]
print(corrected_sentence)
