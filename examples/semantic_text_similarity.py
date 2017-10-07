import numpy as np
import luvina.backend as luv


def preprocess_sentences(sentences, max_token_size=22, remove=False):
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = luv.tokenize(sentence)
        tokens = luv.pad(tokens, max_token_size, remove)
        preprocessed_tokens = []
        for token in tokens:
            preprocessed_token = luv.expand_contractions(token)
            preprocessed_token = luv.get_vector(token)
            preprocessed_tokens.append(preprocessed_token)
        preprocessed_tokens = np.asarray(preprocessed_tokens)
        preprocessed_sentences.append(preprocessed_tokens)
    return np.asarray(preprocessed_sentences)


if __name__ == '__main__':

    from luvina.datasets import semantic_text_similarity
    from luvina.metrics import calculate_r2_score
    from luvina.models import SiameseLSTM

    num_epochs = 100000
    hidden_size = 100
    max_length = 25
    batch_size = 32

    dataset = semantic_text_similarity.load_data()
    input_1 = preprocess_sentences(dataset['Sent1'], max_length)
    input_2 = preprocess_sentences(dataset['Sent2'], max_length)
    output = dataset['Score'].values
    output = output / np.max(output)

    mask = np.random.choice(np.arange(len(output)), len(output), False)
    input_1 = input_1[mask]
    input_2 = input_2[mask]
    output = output[mask]

    model = SiameseLSTM(max_length, hidden_size)
    model.compile('adam', 'mean_squared_error', metrics=[calculate_r2_score])
    model.summary()
    model.fit([input_1, input_2], output, batch_size, num_epochs,
              validation_split=.2)

