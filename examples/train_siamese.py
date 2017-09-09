import numpy as np
from luvina.datasetes import sts
from datasets import sts
from backend import expand_contractions, get_vector, tokenize
from backend import correct_misspelling_ngram

dataset = sts.get_data()


def _preprocess_token(token):
    token = expand_contractions(token)
    token = correct_misspelling_ngram(token)
    token = get_vector(token)
    return token


def _preprocess_sentence(sentence):
    tokens = tokenize(sentence)
    vectors = [_preprocess_token(token) for token in tokens]
    return vectors


def preprocess_sentences(sentences, max_lenghts=25):
    data = []
    for sentence in sentences:
        sentence = _preprocess_sentence(sentence)
        data.append(sentence)
    return data


def _get_length_mask(input_1, input_2, max_length=25):
    lengths_1 = np.asarray([len(tokens) for tokens in input_1])
    lengths_2 = np.asarray([len(tokens) for tokens in input_2])
    mask_1 = lengths_1 < max_length
    mask_2 = lengths_2 < max_length
    mask = np.logical_and(mask_1, mask_2)
    return mask


def _mask_data(data, mask):
    input_1, input_2, output = data
    input_1 = np.asarray(input_1)[mask]
    input_2 = np.asarray(input_2)[mask]
    output = np.asarray(output)[mask]
    masked_data = (input_1.tolist(),
                   input_2.tolist(),
                   output.tolist())
    return masked_data


def _zero_pad(input_data, max_length=25):
    data = []
    for sample in input_data:
        sample = np.asarray(sample)
        sentence_length, embedding_dimension = sample.shape
        missing_zeros = max_length - sentence_length
        zero_array = np.zeros(shape=(missing_zeros, embedding_dimension))
        sample = np.concatenate((sample, zero_array), axis=0)
        data.append(sample)
    return data


def filter_data(data, max_length=25, pad=True):
    input_1, input_2, output = data
    mask = _get_length_mask(input_1, input_2, max_length)
    data = (input_1, input_2, output)
    input_1, input_2, output = _mask_data(data, mask)
    if pad:
        input_1 = _zero_pad(input_1, max_length)
        input_2 = _zero_pad(input_2, max_length)
    return (np.asarray(input_1),
            np.asarray(input_2),
            np.asarray(output))


if __name__ == '__main__':
    from models.siamese import SiameseLSTM
    max_length = 25
    hidden_size = 100
    batch_size = 32
    num_epochs = 100000
    validation_split = .2
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
