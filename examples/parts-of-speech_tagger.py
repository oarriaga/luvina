import luvina.backend as luv

sentence = 'the house is very big and blue'
print(sentence)
tokens = luv.tokenize(sentence)
print(luv.tag_pos(tokens))

