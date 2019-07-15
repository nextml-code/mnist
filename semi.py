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
import data
import architecture


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--unlabeled_fraction', default=0.1, type=float)
    parser.add_argument('--unlabeled_weight', default=0.1, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gradient_clipvalue', default=2.0, type=float)
    parser.add_argument('--sharpen', default=1.0, type=float)
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

    ds_train = data.get_standard_ds(image_train, label_train)
    ds_validate = data.get_standard_ds(image_validate, label_validate)

    model = architecture.get_model(config)
    architecture.compile_model(model, config)

    ds_train_shuffled = data.shuffle_dataset(ds_train, len(label_train))

    sharpen_exponent = 1
    def _sharpen(predicted):
        return data.sharpen(predicted, sharpen_exponent)

    ds_predicted_shuffled = (
        data.shuffle_dataset(ds_validate, len(label_validate))
        .map(lambda image, _: (image, data.augment(image)))
        .batch(config['n_unlabeled'])
        .map(lambda image, augmented_image: (
            image,
            data.predict_batch(model, augmented_image)
        ))
        .map(lambda image, predicted: (
            image,
            _sharpen(predicted)
        ))
        .unbatch()
    )

    ds_fit = data.merge_datasets(
        (
            ds_train_shuffled.map(lambda image, label: (image, label, 1.0)),
            ds_predicted_shuffled.map(lambda image, label: (image, label, config['unlabeled_weight'])),
        ),
        (config['n_labeled'], config['n_unlabeled'])
    ).map(lambda image, label, weight: (
        data.augment(image),
        label,
        weight
    ))

    def update_sharpening(epoch, logs):
        accuracy = logs['val_categorical_accuracy']
        if accuracy <= 0.5:
            sharpen_exponent = 1
        else:
            sharpen_exponent = 2 # np.exp((accuracy - 0.5)*3)
        print(f'update_sharpening sharpen_exponent: {sharpen_exponent:.2f}')

    os.makedirs('checkpoints')
    model.fit(
        ds_fit.batch(config['batch_size']),
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
