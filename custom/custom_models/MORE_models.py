from custom.custom_loss import MORE_loss
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import losses
from tensorflow.keras.regularizers import *
from tensorflow import random


def get_loss(loss):
    loss_dict = {
        "mse": "mse",
        "mae": "mae",
        "corr": MORE_loss.pearson_correlation,
        "z_score_corr": MORE_loss.z_score_pearson_correlation,
        "z_score_mse": MORE_loss.z_score_mse,
        "z_score_mae": MORE_loss.z_score_mae,
        "z_score_huber": MORE_loss.z_score_huber,
        "huber": losses.Huber(delta=1),
        "logcosh": losses.LogCosh,
        "z_score_log_cosh": MORE_loss.z_score_log_cosh,
        "tukeys_biweight": MORE_loss.tukeys_biweight

    }
    return loss_dict[loss]


def get_optimizer(optimizer, lr):
    if optimizer == "Adam":
        return Adam(lr=lr)
    else:
        return RMSprop(lr=lr)


def cnn_np(number_of_samples, loss, hp):
    random.set_seed(hp["seed"])
    input_shape = (number_of_samples, 1)
    input_layer = Input(shape=input_shape, name="input_layer")

    if hp["kernel_regularizer"] == "l1":
        regularizer = l1(l=hp["kernel_regularizer_value"])
    elif hp["kernel_regularizer"] == "l2":
        regularizer = l2(l=hp["kernel_regularizer_value"])
    else:
        regularizer = None

    x = Conv1D(hp['filters'], hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
               kernel_initializer=hp['kernel_initializer'], kernel_regularizer=regularizer, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    for l_i in range(1, hp["conv_layers"]):
        x = Conv1D(hp['filters'] * (l_i + 1), hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
                   kernel_initializer=hp['kernel_initializer'], kernel_regularizer=regularizer, padding='same')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    x = Flatten()(x)

    x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['kernel_initializer'],
              kernel_regularizer=regularizer, name=f'fc_1')(x)
    for l_i in range(1, hp["layers"]):
        x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['kernel_initializer'],
                  kernel_regularizer=regularizer, name=f'fc_{l_i + 1}')(x)
        if hp["dropout"] > 0:
            x = Dropout(rate=hp["dropout"])(x)

    output_layer = Dense(256, activation='linear', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='cnn_search')
    m_model.compile(loss=get_loss(loss), optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]))
    m_model.summary()
    return m_model


def mlp_np(number_of_samples, loss, hp):
    random.set_seed(hp["seed"])

    input_layer = Input(shape=(number_of_samples,), name="input_layer")

    if hp["kernel_regularizer"] == "l1":
        regularizer = l1(l=hp["kernel_regularizer_value"])
    elif hp["kernel_regularizer"] == "l2":
        regularizer = l2(l=hp["kernel_regularizer_value"])
    else:
        regularizer = None

    x = None
    for l_i in range(hp["layers"]):
        x = Dense(hp["neurons"],
                  activation=hp["activation"],
                  kernel_initializer=hp["kernel_initializer"],
                  kernel_regularizer=regularizer,
                  name=f"layer_{l_i}")(
            input_layer if l_i == 0 else x)
        if hp["dropout"] > 0:
            x = Dropout(rate=hp["dropout"])(x)
    output = Dense(256, activation="linear", name='scores')(x)

    m_model = Model(input_layer, output, name='mlp_search')
    m_model.compile(loss=get_loss(loss), optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]))
    m_model.summary()
    return m_model
