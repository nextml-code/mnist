import tensorflow as tf
import numpy as np
import sklearn
import sklearn.model_selection

IMAGE_SHAPE = [28, 28]

TRAIN = 'train'
VALIDATE = 'validate'
TEST = 'test'


def _get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    index_train, index_validate = sklearn.model_selection.train_test_split(
        np.arange(len(x_train)),
        stratify=y_train,
        test_size=0.999,
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


def get_data(data_type):
    return _get_data()[data_type]


def get_score(y_score, data_type):
    x, y = _get_data()[data_type]
    return sklearn.metrics.accuracy_score(
        y,
        y_score,
    )
