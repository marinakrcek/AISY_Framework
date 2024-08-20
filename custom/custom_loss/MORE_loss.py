import tensorflow.keras.backend as K
import tensorflow as tf


def pearson_correlation(y_true, y_pred):
    x = K.cast(y_true, dtype="float32")
    y = K.cast(y_pred, dtype="float32")
    mx = tf.reduce_mean(x)
    my = tf.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.sqrt(tf.reduce_mean(tf.square(xm))) * tf.math.sqrt(tf.reduce_mean(tf.square(ym)))
    r = r_num / r_den
    r = tf.where(tf.math.is_nan(r), 0., r)
    return 1 - tf.abs(r)


def z_score_pearson_correlation(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    mx = tf.reduce_mean(z_true)
    my = tf.reduce_mean(z_pred)
    xm, ym = z_true - mx, z_pred - my
    r_num = tf.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.sqrt(tf.reduce_mean(tf.square(xm))) * tf.math.sqrt(tf.reduce_mean(tf.square(ym)))
    r = r_num / r_den
    r = tf.where(tf.math.is_nan(r), 0., r)
    return 1 - tf.abs(r)


def z_score_mse(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    return tf.reduce_mean(tf.math.square(z_true - z_pred))


def z_score_mae(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    return tf.reduce_mean(tf.math.abs(z_true - z_pred))


def tukeys_biweight(y_true, y_pred):
    delta = 4.5
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    error = y_true - y_pred
    abs_error = tf.abs(error)

    error_sq = (y_true - y_pred) ** 2
    mask_below = tf.cast((abs_error <= delta), tf.float32)
    rho_above = tf.cast((abs_error > delta), tf.float32) * delta / 2

    rho_below = (delta / 2) * (1 - ((1 - ((error_sq * mask_below) / delta)) ** 3))
    rho = rho_above + rho_below

    return tf.reduce_mean(rho)


def log_cosh(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    error = y_true - y_pred
    abs_error = tf.math.log(tf.cosh(error))
    loss = abs_error
    return tf.reduce_sum(loss)


def z_score_log_cosh(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    error = z_true - z_pred
    abs_error = tf.math.log(tf.cosh(error))
    loss = abs_error
    return tf.reduce_sum(loss)


def huber(y_true, y_pred):
    delta = 1.0
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(loss)


def z_score_huber(y_true, y_pred):
    y_true = K.cast(y_true, dtype="float32")
    y_pred = K.cast(y_pred, dtype="float32")
    y_true_mean = tf.math.reduce_mean(y_true)
    y_true_std = tf.math.reduce_std(y_true)
    y_pred_mean = tf.math.reduce_mean(y_pred)
    y_pred_std = tf.math.reduce_std(y_pred)

    z_true = (y_true - y_true_mean) / tf.maximum(y_true_std, 1e-10)
    z_pred = (y_pred - y_pred_mean) / tf.maximum(y_pred_std, 1e-10)

    delta = 1
    error = z_true - z_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic ** 2 + delta * linear
    return tf.reduce_mean(loss)