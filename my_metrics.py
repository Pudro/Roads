import tensorflow as tf


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_true, tf.int32)
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_true, tf.int32)
    smooth = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    intersection = tf.cast(intersection, tf.float32)
    union = tf.cast(intersection, tf.float32)
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou

def dice_dice_loss(y_true, y_pred):
    GAMMA = 2
    SMOOTH = 1e-6

    y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
    nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + SMOOTH
    denominator = tf.reduce_sum(y_pred ** GAMMA) + tf.reduce_sum(y_true ** GAMMA) + SMOOTH
    result = 1 - tf.divide(nominator, denominator)
    return result

