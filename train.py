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
            l.Dense(32, activation=tf.nn.relu),
            # l.Dropout(0.4),
            l.Dense(10, activation=tf.nn.softmax)
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


def augment(ds, image_augmentor):
    return ds.map(lambda image, label: (
        tf.numpy_function(
            func=image_augmentor.random_transform,
            inp=[image],
            Tout=[tf.float32]
        )[0],
        label
    ))


def shuffle_dataset(ds, buffer_size):
    return (
        ds
        .repeat()
        .shuffle(
            buffer_size,
            reshuffle_each_iteration=True,
            # seed=seed
        )
    )


def predict_dataset(ds, model, batch_size=60):
    return (
        ds
        .batch(batch_size)
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
        # .map(lambda image, predicted: (
        #     image,
        #     tf.cast(
        #         tf.equal(predicted, tf.reduce_max(predicted, axis=-1, keepdims=True)),
        #         tf.float32
        #     )
        # ))
        .unbatch()
    )


def merge_datasets(ds_a, ds_b, n_a, n_b):
    return (
        tf.data.Dataset.zip((
            ds_a.batch(n_a),
            ds_b.batch(n_b)
        ))
        .flat_map(lambda batch_a, batch_b: (
            tf.data.Dataset.from_tensors(batch_a)
            .unbatch()
            .concatenate(
                tf.data.Dataset.from_tensors(batch_b)
                .unbatch()
            )
        ))
    )

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--steps_per_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--n_labeled', default=1, type=int)
    parser.add_argument('--n_unlabeled', default=1, type=int)
    parser.add_argument('--batch_size', default=120, type=int)
    parser.add_argument('--supervised_epochs', default=5, type=int)
    parser.add_argument('--gradient_clipvalue', default=1.0, type=float)
    parser.add_argument('--seed', default=np.random.randint(1000), type=int)

    args = parser.parse_args()
    config = vars(args)
    config.update(
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
        shear_range=30,
        zoom_range=[0.6, 1.2],
        # brightness_range=[0.1, 1],
    )

    # import matplotlib.pyplot as plt
    # image = np.zeros((28, 28, 1))
    # image[5:23, 5:23] = 1.0
    # plt.imshow(image[:,:,0], cmap='gray')
    # plt.imshow(image_augmentor.random_transform(image)[:,:,0], cmap='gray', vmax=255.0)

    ds_train_shuffled = shuffle_dataset(ds_train, len(label_train))

    ds_predicted_shuffled = predict_dataset(
        shuffle_dataset(ds_validate, len(label_validate)), model
    )

    # next(iter(ds_train.batch(10)))[1].shape
    # next(iter(ds_unsupervised))[0].shape

    # import matplotlib.pyplot as plt
    # plt.imshow(next(iter(ds_predicted_shuffled.skip(2)))[0][:,:,0], cmap='gray', vmax=1)

    ds_semisupervised = merge_datasets(
        ds_train_shuffled,
        ds_predicted_shuffled,
        config['n_labeled'],
        config['n_unlabeled']
    )

    # it = iter(augment(ds_semisupervised, image_augmentor))
    # import matplotlib.pyplot as plt
    # plt.imshow(next(it)[0][:,:,0], cmap='gray', vmax=1.0)

    if config['supervised_epochs'] >= 1:
        model.fit(
            augment(ds_train_shuffled, image_augmentor).batch(60),
            validation_data=ds_validate.batch(1024*4),
            epochs=config['supervised_epochs'],
            steps_per_epoch=100,
            verbose=1
        )


    os.makedirs('checkpoints')
    model.fit(
        # augment(ds_train_shuffled, image_augmentor).batch(config['batch_size']),
        augment(ds_semisupervised, image_augmentor).batch(config['batch_size']),
        # ds_unsupervised.batch(64),
        validation_data=ds_validate.batch(1024*4),
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
            keras.callbacks.EarlyStopping(
                monitor='val_categorical_accuracy',
                min_delta=1e-2,
                patience=10,
                verbose=1
            ),
        ],
        verbose=1
    )

# [ ] verify model changes
# [ ] prefetch lots of predictions
# [ ] predict full dataset then run X epochs, repeat
# [x] augmentation unsupervised
# [x] augmentation on training too
# [ ] handle overfitting before sharpen
# [ ] noisy network
