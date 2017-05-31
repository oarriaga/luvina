import luvina.backend as luv

def preprocess_sentence(sentence):
    tokens = luv.tokenize(sentence)
    tokens = luv.filter_stop_words(tokens)
    tokens = luv.filter_repeated_words(tokens)
    return tokens

sentence_1 = 'there is a dog and the dog sat on the mat'
sentence_2 = 'there is a mat and on the mat sat the hound'

tokens_1 = preprocess_sentence(sentence_1)
tokens_2 = preprocess_sentence(sentence_2)

synonyms = luv.find_synonyms(tokens_1, tokens_2)
print(synonyms)

edited_sentence = luv.edit(luv.tokenize(sentence_2), synonyms)
print(edited_sentence)

