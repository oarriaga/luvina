from keras.models import Model
from keras.layers import Input, Masking, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.merge import Dot


def SiameseLSTM(max_token_length, hidden_size, embedding_size=300):
    text_input_1 = Input(shape=(max_token_length, embedding_size),
                         name='text_1')
    text_mask_1 = Masking(mask_value=0.0, name='text_mask_1')(text_input_1)
    text_dropout_1 = Dropout(.5, name='text_dropout')(text_mask_1)

    text_input_2 = Input(shape=(max_token_length, embedding_size),
                         name='text_2')
    text_mask_2 = Masking(mask_value=0.0, name='text_mask_2')(text_input_2)
    text_dropout_2 = Dropout(.5, name='text_dropout')(text_mask_2)

    lstm_1 = LSTM(units=hidden_size,
                  return_sequences=False,
                  name='lstm_1')(text_dropout_1)

    lstm_2 = LSTM(units=hidden_size,
                  return_sequences=False,
                  name='lstm_2')(text_dropout_2)

    cosine_similarity = Dot(axis=1, normalize=True,
                            name='cosine_similarity')([lstm_1, lstm_2])

    model = Model(inputs=[text_input_1, text_input_2],
                  outputs=cosine_similarity)

    return model
