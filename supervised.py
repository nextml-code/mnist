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

    # parser.add_argument('--steps_per_epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gradient_clipvalue', default=2.0, type=float)
    parser.add_argument('--seed', default=np.random.randint(1000), type=int)

    args = parser.parse_args()
    config = vars(args)
    config.update(
        max_epochs=100,
    )

    image_train, label_train = problem.get_data(problem.TRAIN)
    image_validate, label_validate = problem.get_data(problem.VALIDATE)

    ds_train = data.get_standard_ds(image_train, label_train)
    ds_validate = data.get_standard_ds(image_validate, label_validate)

    model = architecture.get_model(config)
    architecture.compile_model(model, config)

    ds_train_shuffled = data.shuffle_dataset(ds_train, len(label_train))

    os.makedirs('checkpoints')
    model.fit(
        ds_train_shuffled.map(lambda image, label: (
            data.augment(image),
            label
        )).batch(config['batch_size']),
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
        ],
        verbose=1
    )

    result = model.evaluate(ds_validate.batch(1024*4), verbose=0)
    result = dict(zip(model.metrics_names, result))
    print(f'val_categorical_accuracy: {result["categorical_accuracy"]}')
