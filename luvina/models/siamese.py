from keras.models import Model
from keras.layers import Input, Masking
from keras.layers.recurrent import GRU
from keras.layers.merge import Dot
from keras.layers.wrappers import Bidirectional


def SiameseLSTM(max_token_length, hidden_size, embedding_size=300):
    text_input_1 = Input(shape=(max_token_length, embedding_size),
                         name='text_1')
    text_mask_1 = Masking(mask_value=0.0, name='text_mask_1')(text_input_1)
    # text_dropout_1 = Dropout(.5, name='text_dropout_1')(text_mask_1)

    text_input_2 = Input(shape=(max_token_length, embedding_size),
                         name='text_2')
    text_mask_2 = Masking(mask_value=0.0, name='text_mask_2')(text_input_2)
    # text_dropout_2 = Dropout(.5, name='text_dropout_2')(text_mask_2)

    lstm_1_a = Bidirectional(GRU(units=hidden_size,
                                 return_sequences=True,
                                 name='RNN_1_a'))(text_mask_1)

    lstm_1_b = Bidirectional(GRU(units=hidden_size,
                                 return_sequences=False,
                                 name='RNN_1_b'))(lstm_1_a)

    """
    lstm_1_c = Bidirectional(GRU(units=hidden_size,
                                 return_sequences=False,
                                 name='RNN_1_c'))(lstm_1_b)
    """

    lstm_2_a = Bidirectional(GRU(units=hidden_size,
                                 return_sequences=True,
                                 name='RNN_2_a'))(text_mask_2)

    lstm_2_b = Bidirectional(GRU(units=hidden_size,
                                 return_sequences=False,
                                 name='RNN_2_b'))(lstm_2_a)

    """
    lstm_2_c = Bidirectional(GRU(units=hidden_size,
                                 return_sequences=False,
                                 name='RNN_2_c'))(lstm_2_b)
    """

    cosine_similarity = Dot(axes=1, normalize=True,
                            name='cosine_similarity')([lstm_1_b, lstm_2_b])

    model = Model(inputs=[text_input_1, text_input_2],
                  outputs=cosine_similarity)

    return model


if __name__ == '__main__':
    model = SiameseLSTM(25, 50)
    model.summary()
