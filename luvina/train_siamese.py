import numpy as np
from datasets import sts
from backend import expand_contractions, get_vector, tokenize

dataset = sts.get_data()


def preprocess_token(token):
    token = expand_contractions(token)
    # token = correct_misspelling(token)
    token = get_vector(token)
    return token


def preprocess_sentence(sentence):
    tokens = tokenize(sentence)
    vectors = [preprocess_token(token) for token in tokens]
    return vectors


def preprocess_sentences(sentences, max_lenghts=25):
    data = []
    for sentence in sentences:
        sentence = preprocess_sentence(sentence)
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
    max_length = 25
    input_1 = preprocess_sentences(dataset['Sent1'])
    input_2 = preprocess_sentences(dataset['Sent2'])
    output = dataset['Score'].values
    data = (input_1, input_2, output)
    input_1, input_2, output = filter_data(data, max_length)
