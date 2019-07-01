import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow.keras as keras
import numpy as np
import os
import argparse

import problem


def get_model():
    l = tf.keras.layers

    input = l.Input(problem.IMAGE_SHAPE + [1], name='image')

    max_pool = l.MaxPooling2D((2, 2), padding='same')

    probabilities = tf.keras.Sequential(
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
            l.Dense(32, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
            # l.Dropout(0.4),
            l.Dense(10, activation=tf.nn.softmax, kernel_regularizer=keras.regularizers.l2(0.001))
        ],
        name='probabilities'
    )(input)

    return keras.models.Model(inputs=input, outputs=probabilities)


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


def preprocess_image(image):
    image = image / 255.0
    image = np.expand_dims(image, -1)
    return image.astype(np.float32)


def to_one_hot(label):
    return np.eye(10)[label]


def preprocess_label(label):
    return to_one_hot(label).astype(np.float32)


def get_standard_ds(image, label):
    return (
        tf.data.Dataset.from_tensor_slices((
            preprocess_image(image), preprocess_label(label)
        ))
    )

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--max_epochs', default=100, type=int)
    # parser.add_argument('--steps_per_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    # parser.add_argument('--batch_size', default=100, type=int)
    # parser.add_argument('--gradient_clipvalue', default=1.0, type=float)
    # parser.add_argument('--seed', default=71751, type=int)

    args = parser.parse_args()
    config = vars(args)
    config.update(
        gradient_clipvalue=1,
        max_epochs=100,
    )

    image_train, label_train = problem.get_data(problem.TRAIN)
    image_validate, label_validate = problem.get_data(problem.VALIDATE)

    ds_train = get_standard_ds(image_train, label_train)
    ds_validate = get_standard_ds(image_validate, label_validate)

    model = get_model()
    compile_model(model, config)

    image_augmentor = keras.preprocessing.image.ImageDataGenerator(
        fill_mode='constant',
        cval=0.0,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=45,
        zoom_range=[0.6, 1.2],
    )

    ds_unsupervised = (
        ds_validate
        .repeat()
        .shuffle(
            buffer_size=len(label_validate),
            reshuffle_each_iteration=True,
            # seed=seed
        )
        .batch(60)
        .map(lambda image, _: (
            image,
            tf.py_function(
                func=model.predict_on_batch,
                inp=[image],
                Tout=tf.float32,
            )
        ))
        # .map(lambda image, predicted: (
        #     image,
        #     predicted**2 / tf.reduce_sum(predicted**2, axis=-1, keepdims=True)
        # ))
        .map(lambda image, predicted: (
            image,
            tf.cast(
                tf.equal(predicted, tf.reduce_max(predicted, axis=-1, keepdims=True)),
                tf.float32
            )
        ))
        .unbatch()
        .map(lambda image, predicted: (
            tf.numpy_function(
                func=image_augmentor.random_transform,
                inp=[image],
                Tout=[tf.float32]
            )[0],
            predicted
        ))
        # .map(lambda image, predicted: (
        #     tf.numpy_function(
        #         func=lambda image: keras.preprocessing.image.ImageDataGenerator(
        #             fill_mode='constant',
        #             cval=0.0,
        #         ).apply_transform(
        #             image,
        #             dict(
        #                 theta=np.random.uniform(-10, 10)/180*np.pi, # Rotation angle in degrees.
        #                 # tx=np.random.uniform(-0.25, 0.25)*28, # Shift in the x direction.
        #                 # ty=np.random.uniform(-0.25, 0.25)*28, # Shift in the y direction.
        #             )
        #         ),
        #         inp=[image],
        #         Tout=[tf.float32]
        #     ),
        #     predicted
        # ))
    )

    # next(iter(ds_train.batch(10)))[1].shape
    # next(iter(ds_unsupervised))[0].shape

    # import matplotlib.pyplot as plt
    # plt.imshow(next(iter(ds_unsupervised.skip(2)))[0][:,:,0])

    ds_semisupervised = (
        tf.data.Dataset.zip((
            (
                ds_train
                .repeat()
                .shuffle(
                    buffer_size=len(label_validate),
                    reshuffle_each_iteration=True,
                    # seed=seed
                )
                .batch(1)
            ),
            ds_unsupervised.batch(1)
        ))
        .flat_map(
            lambda labeled_batch, unsupervised_batch: (
                tf.data.Dataset.from_tensors(labeled_batch)
                .unbatch()
                .concatenate(
                    tf.data.Dataset.from_tensors(unsupervised_batch)
                    .unbatch()
                )
            )
        )
    )

    # next(iter(ds_semisupervised.batch(10)))[1].shape

    model.fit(
        (
            ds_train
            .repeat()
            .shuffle(
                buffer_size=len(label_train),
                reshuffle_each_iteration=True,
                # seed=seed
            )
            .batch(60)
        ),
        validation_data=ds_validate.batch(1024),
        epochs=5,
        steps_per_epoch=100,
        verbose=1
    )


    os.makedirs('checkpoints')
    model.fit(
        # (
        #     ds_train
        #     .repeat()
        #     .shuffle(
        #         buffer_size=len(label_train),
        #         reshuffle_each_iteration=True,
        #         # seed=seed
        #     )
        #     .batch(60)
        # ),
        ds_semisupervised.batch(120),
        # ds_unsupervised.batch(64),
        validation_data=ds_validate.batch(1024),
        # batch_size=1024,
        epochs=config['max_epochs'],
        steps_per_epoch=100,
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

# [ ] verify model changes
# [ ] prefetch lots of predictions
# [ ] predict full dataset then run X epochs, repeat
# [ ] augmentation
