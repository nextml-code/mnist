import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

import problem


def get_model():
    l = tf.keras.layers

    input = l.Input(problem.IMAGE_SHAPE + [1], name='image')

    max_pool = l.MaxPooling2D((2, 2), padding='same')

    logits = tf.keras.Sequential(
        [
            l.Conv2D(64, kernel_size=5, padding='same', activation=tf.nn.relu),
            max_pool,
            l.Conv2D(32, kernel_size=5, padding='same', activation=tf.nn.relu),
            max_pool,
            l.Conv2D(16, kernel_size=3, padding='same', activation=tf.nn.relu),
            max_pool,
            l.Conv2D(8, kernel_size=3, padding='same', activation=tf.nn.relu),
            max_pool,
            l.Flatten(),
            l.Dense(32, activation=tf.nn.relu),
            # l.Dropout(0.4),
            l.Dense(10)
        ],
        name='logits'
    )(input)

    return keras.models.Model(inputs=input, outputs=logits)


def compile_model(model, config):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            clipvalue=config['gradient_clipvalue'],
        ),
        loss=dict(
            logits=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        ),
        metrics=dict(
            logits=[
                tf.keras.metrics.SparseCategoricalAccuracy()
            ],
        ),
        target_tensors=dict(
            logits=keras.layers.Input([1]),
        ),
        # run_eagerly=True,
    )


def preprocess(x):
    x = x / 255.0
    x = np.expand_dims(x, -1)
    return x

# %%
if __name__ == '__main__':
    config = dict(
        learning_rate=0.01,
        gradient_clipvalue=1,
        max_epochs=100,
    )

    x_train, y_train = problem.get_data(problem.TRAIN)
    x_validate, y_validate = problem.get_data(problem.VALIDATE)

    model = get_model()
    compile_model(model, config)

    os.makedirs('checkpoints')
    model.fit(
        x=preprocess(x_train),
        y=y_train,
        validation_data=(
            preprocess(x_validate),
            y_validate
        ),
        batch_size=1024,
        epochs=config['max_epochs'],
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir='tb',
                update_freq='batch',
                histogram_freq=0,
                write_graph=True,
                write_images=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/epoch{epoch}.h5',
                # save_best_only=True,
                save_weights_only=True,
                # monitor='val_loss',
                verbose=1
            ),
            # keras.callbacks.EarlyStopping(
            #     monitor='val_loss',
            #     min_delta=1e-2,
            #     patience=10,
            #     verbose=1
            # ),
        ],
        verbose=1
    )
