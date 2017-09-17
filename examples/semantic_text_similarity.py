import numpy as np
import luvina.backend as luv


def preprocess_sentences(sentences, max_token_size, remove):
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = luv.tokenize(sentence)
        tokens = luv.pad(tokens, max_token_size, remove)
        preprocessed_tokens = []
        for token in tokens:
            preprocessed_token = luv.expand_contractions(token)
            preprocessed_token = luv.correct_misspelling(token)
            preprocessed_token = luv.get_vector(token)
            preprocessed_tokens.append(preprocessed_token)
        preprocessed_sentences.append(preprocessed_tokens)
    return np.concatenate(preprocess_sentences, axis=0)


def 2preprocess_sentences(sentences):
    sentences = [luv.pad(luv.tokenize(sentence)) for sentence in sentences]


def preprocess_sentence(sentence):
    tokens = luv.tokenize(sentence)
    tokens = luv.pad(tokens)
    vectors = [preprocess_token(token) for token in tokens]
    return vectors


def preprocess_token(token):
    token = luv.expand_contractions(token)
    token = luv.correct_misspelling(token)
    token = luv.get_vector(token)
    return token


if __name__ == '__main__':

    from luvina.models import SiameseLSTM
    from luvina.datasets import semantic_text_similarity

    max_length = 25
    hidden_size = 100
    batch_size = 32
    num_epochs = 100000
    validation_split = .2

    dataset = semantic_text_similarity.load_data()
    input_1 = preprocess_sentences(dataset['Sent1'])
    input_2 = preprocess_sentences(dataset['Sent2'])
    output = dataset['Score'].values
    data = (input_1, input_2, output)
    input_1, input_2, output = filter_data(data, max_length)
    output = output / np.max(output)

    """
    model = SiameseLSTM(max_length, hidden_size)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    model.fit([input_1, input_2], output, batch_size, num_epochs,
              validation_split=validation_split)
    """
