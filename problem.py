import tensorflow as tf
import numpy as np
import sklearn
import sklearn.model_selection

IMAGE_SHAPE = [28, 28]

TRAIN = 'train'
VALIDATE = 'validate'
TEST = 'test'


def _get_data(train_size=None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if train_size is None:
        train_size = 0.0002

    index_train, index_validate = sklearn.model_selection.train_test_split(
        np.arange(len(x_train)),
        stratify=y_train,
        test_size=1 - train_size,
        random_state=1,
    )

    return dict(
        train=(
            x_train[index_train],
            y_train[index_train],
        ),
        validate=(
            x_train[index_validate],
            y_train[index_validate],
        ),
        test=(
            x_test,
            y_test,
        )
    )


def training_count(train_size=None):
    return np.unique(_get_data(train_size)['train'][1], return_counts=True)


def get_data(data_type, train_size=None):
    return _get_data(train_size)[data_type]


def get_score(y_score, data_type, train_size=None):
    x, y = _get_data(train_size)[data_type]
    return sklearn.metrics.accuracy_score(
        y,
        y_score,
    )
