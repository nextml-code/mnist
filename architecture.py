import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import os
import argparse

import problem


def get_model(config):
    l = tf.keras.layers

    image = l.Input(problem.IMAGE_SHAPE + [1], name='image')

    max_pool = l.MaxPooling2D((2, 2), padding='same')

    probabilities = tf.keras.Sequential(
        [
            l.Conv2D(20, kernel_size=5, padding='same', activation=tf.nn.relu),
            max_pool,
            l.Conv2D(50, kernel_size=5, padding='same', activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(100, activation=tf.nn.relu),
            l.Dense(10, activation=tf.nn.softmax)
        ],
        name='probabilities'
    )(image)

    return keras.models.Model(inputs=image, outputs=probabilities)


def compile_model(model, config):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            clipvalue=config['gradient_clipvalue'],
        ),
        loss=dict(
            probabilities=tf.keras.losses.CategoricalCrossentropy(),
            # probabilities=tf.keras.losses.MeanSquaredError(),
        ),
        metrics=dict(
            probabilities=[
                tf.keras.metrics.CategoricalAccuracy()
            ],
        ),
        target_tensors=dict(
            probabilities=keras.layers.Input([1]),
        ),
        # run_eagerly=True,
    )
