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

    input = l.Input(problem.IMAGE_SHAPE + [1], name='image')

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


def augment(image, image_augmentor):
    return tf.reshape(tf.numpy_function(
        func=image_augmentor.random_transform,
        inp=[image],
        Tout=[tf.float32]
    )[0], problem.IMAGE_SHAPE + [1])


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


def sharpen(probs, exponent):
    p = probs**exponent
    return p / tf.reduce_sum(p, axis=-1, keepdims=True)


def argmax_sharpen():
    return tf.cast(
        tf.equal(predicted, tf.reduce_max(predicted, axis=-1, keepdims=True)),
        tf.float32
    )


def predict_batch(image):
    return tf.py_function(
        func=model.predict_on_batch,
        inp=[image],
        Tout=tf.float32,
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


def get_beta_weights(shape, a, b):
    # p = tfp.distributions.Beta(a, b).sample(shape)
    p = 1 - tf.abs(tfp.distributions.Beta(0.5, 0.5).sample(shape))
    return tf.concat([p, (1 - p)], axis=0)

# tfp.distributions.Beta(a, b).sample(shape)
# tf.einsum('i...,i->...', tf.ones((2, 28, 28, 1)), tf.ones((2,))/5)

def mixup_items(items, weight_func):
    weights = weight_func(1)
    weights /= tf.reduce_sum(weights)

    return tuple([
        tf.einsum('i...,i->...', tf.stack(variable, axis=0), weights)
        for variable in zip(*items)
    ])


def mixup_datasets(datasets, weight_func):
    return (
        tf.data.Dataset.zip(datasets)
        .map(lambda *items: mixup_items(items, weight_func))
    )

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--steps_per_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--unlabeled_fraction', default=0.1, type=float)
    parser.add_argument('--unlabeled_weight', default=0.1, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--supervised_epochs', default=0, type=int)
    parser.add_argument('--gradient_clipvalue', default=2.0, type=float)
    parser.add_argument('--sharpen', default=1.0, type=float)
    parser.add_argument('--noisy_weight', default=1.0, type=float)
    parser.add_argument('--seed', default=np.random.randint(1000), type=int)

    args = parser.parse_args()
    config = vars(args)
    config.update(
        max_epochs=100,
        n_labeled=int(np.ceil(config['unlabeled_fraction']*config['batch_size'])),
        n_unlabeled=int(np.floor(config['unlabeled_fraction']*config['batch_size'])),
    )

    image_train, label_train = problem.get_data(problem.TRAIN)
    image_validate, label_validate = problem.get_data(problem.VALIDATE)

    ds_train = get_standard_ds(image_train, label_train)
    ds_validate = get_standard_ds(image_validate, label_validate)

    model = get_model(config)
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

    sharpen_exponent = 1
    def _sharpen(predicted):
        return sharpen(predicted, sharpen_exponent)

    if config['n_unlabeled'] == 0:
        ds_semisupervised = ds_train_shuffled.map(lambda image, label: (
            augment(image, image_augmentor),
            label
        ))
    else:
        ds_predicted_shuffled = (
            shuffle_dataset(ds_validate, len(label_validate))
            .map(lambda image, _: (image, augment(image, image_augmentor)))
            .batch(config['n_unlabeled'])
            .map(lambda image, augmented_image: (
                image,
                predict_batch(augmented_image)
            ))
            .map(lambda image, predicted: (
                image,
                _sharpen(predicted)
            ))
            .unbatch()
        )

        # next(iter(ds_train.batch(10)))[1].shape
        # next(iter(ds_unsupervised))[0].shape

        # import matplotlib.pyplot as plt
        # plt.imshow(next(iter(ds_predicted_shuffled.skip(2)))[0][:,:,0], cmap='gray', vmax=1)

        ds_semisupervised = mixup_datasets(
            (
                ds_train_shuffled.map(lambda image, label: (
                    augment(image, image_augmentor),
                    label
                )),
                ds_predicted_shuffled.map(lambda image, label: (
                    augment(image, image_augmentor),
                    label
                ))
            ),
            lambda shape: get_beta_weights(shape, a=8, b=1)
        )

        # ds_semisupervised = merge_datasets(
        #     ds_train_shuffled.map(lambda image, label: (image, label, 1.0)),
        #     ds_predicted_shuffled.map(lambda image, label: (image, label, 0.1)),
        #     config['n_labeled'],
        #     config['n_unlabeled']
        # )

    it = iter(ds_semisupervised)
    import matplotlib.pyplot as plt
    next(it)[1].shape
    plt.imshow(next(it)[0][:,:,0], cmap='gray', vmax=1.0)
    plt.show()

    if config['supervised_epochs'] >= 1:
        model.fit(
            ds_train_shuffled.map(lambda image, label: (
                augment(image, image_augmentor),
                label
            )).batch(60),
            validation_data=ds_validate.batch(1024*4),
            epochs=config['supervised_epochs'],
            steps_per_epoch=100,
            verbose=1
        )

    def update_sharpening(epoch, logs):
        accuracy = logs['val_categorical_accuracy']
        if accuracy <= 0.5:
            sharpen_exponent = 1
        else:
            sharpen_exponent = 2 # np.exp((accuracy - 0.5)*3)
        print(f'update_sharpening sharpen_exponent: {sharpen_exponent:.2f}')

    os.makedirs('checkpoints')
    model.fit(
        # ds_train_shuffled.map(lambda image, label: (
        #     augment(image, image_augmentor),
        #     label
        # )).batch(config['batch_size']),
        ds_semisupervised.batch(config['batch_size']),
        # ds_semisupervised.map(lambda image, label, weight: (
        #     augment(image, image_augmentor),
        #     label,
        #     weight
        # )).batch(config['batch_size']),
        validation_data=ds_validate.batch(1024*4),
        epochs=config['max_epochs'],
        steps_per_epoch=10,
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir='tb',
                update_freq='batch',
                histogram_freq=0,
                # write_graph=True,
                # write_images=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/epoch{epoch}.h5',
                # save_best_only=True,
                save_weights_only=True,
                # monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_categorical_accuracy',
                mode='max',
                factor=0.2,
                patience=10,
                min_lr=0.00001,
                verbose=1
            ),
            # keras.callbacks.EarlyStopping(
            #     monitor='val_categorical_accuracy',
            #     mode='max',
            #     min_delta=1e-2,
            #     patience=15,
            #     verbose=1,
            #     restore_best_weights=True
            # ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=update_sharpening
            )
        ],
        verbose=1
    )

    result = model.evaluate(ds_validate.batch(1024*4), verbose=0)
    result = dict(zip(model.metrics_names, result))
    print(f'val_categorical_accuracy: {result["categorical_accuracy"]}')


# [ ] verify model changes
# [ ] prefetch lots of predictions
# [ ] predict full dataset then run X epochs, repeat
# [x] augmentation unsupervised
# [x] augmentation on training too
# [ ] handle overfitting before sharpen
# [ ] noisy network
# [ ] "Focal loss", focus on labeled
# [ ] More augmentations
