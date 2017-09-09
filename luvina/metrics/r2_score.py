import keras.backend as K


def calculate_r2_score(y_true, y_pred):
    """ calculates the R2 score
    args:
        y_true: keras tensor
        y_pred: keras tensor
    returns:
        R2: tensor
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    R2 = 1 - (SS_res / (SS_tot + K.epsilon()))
    return R2
