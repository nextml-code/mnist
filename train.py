import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import os
import random
import argparse

import problem
import data
import architecture


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--steps_per_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--gradient_clipvalue', default=2.0, type=float)
    parser.add_argument('--n_predictions', default=4, type=int)
    parser.add_argument('--sharpen_exponent', default=6.0, type=float)
    parser.add_argument('--n_supervised', default=0, type=int)
    parser.add_argument('--n_unsupervised', default=0, type=int)
    parser.add_argument('--n_mixup_supervised', default=0, type=int)
    parser.add_argument('--n_mixup_semisupervised', default=1, type=int)
    parser.add_argument('--n_mixup_semisupervised_reverse', default=0, type=int)
    parser.add_argument('--n_mixup_unsupervised', default=0, type=int)
    parser.add_argument('--problem_train_size', default=0.0002, type=float)
    parser.add_argument('--seed', default=np.random.randint(1000), type=int)

    args = parser.parse_args()
    config = vars(args)
    config.update(
        evaluation_batch_size=1024*4
    )

    np.random.seed(seed=config['seed'])
    random.seed(np.random.randint(config['seed']))
    tf.random.set_seed(np.random.randint(config['seed']))

    image_train, label_train = problem.get_data(
        problem.TRAIN,
        train_size=config['problem_train_size']
    )
    image_validate, label_validate = problem.get_data(
        problem.VALIDATE,
        train_size=config['problem_train_size']
    )
    image_test, label_test = problem.get_data(
        problem.TEST,
        train_size=config['problem_train_size']
    )

    ds_train = data.get_standard_ds(image_train, label_train)
    ds_validate = data.get_standard_ds(image_validate, label_validate)
    ds_test = data.get_standard_ds(image_test, label_test)

    model = architecture.get_model(config)
    architecture.compile_model(model, config)

    if os.path.exists('model'):
        print('Loading model checkpoint')
        model.load_weights(os.path.join('model', 'checkpoints', 'best_weights.h5'))

    ds_train_shuffled = data.shuffle_dataset(ds_train, len(label_train))
    ds_validate_shuffled = data.shuffle_dataset(ds_validate, len(label_validate))

    ds_predictions_shuffled = (
        ds_validate_shuffled
        .batch(1)
        .flat_map(lambda *batch: (
            tf.data.Dataset.from_tensors(batch).repeat(config['n_predictions']).unbatch()
        ))
        .map(lambda image, _: data.augment(image))
        .batch(config['batch_size']*config['n_predictions'])
        .map(lambda image: data.predict_batch(model, image))
        .unbatch()
        .map(lambda prediction: data.sharpen(prediction, config['sharpen_exponent'])) # TODO: should sharpen after mean
        .batch(config['n_predictions'])
        .map(lambda predictions: tf.reduce_mean(predictions, axis=0))
    )

    ds_unsupervised_shuffled = tf.data.Dataset.zip((
        ds_validate_shuffled.map(lambda image, _: image),
        ds_predictions_shuffled
    ))

    ds_fit_supervised = ds_train_shuffled.map(lambda image, label: (
        data.augment(image), label
    ))

    ds_fit_unsupervised = ds_unsupervised_shuffled.map(lambda image, label: (
        data.augment(image), label
    ))

    ds_fit = data.merge_datasets(
        (
            ds_fit_supervised,
            ds_fit_unsupervised,
            data.mixup_datasets((ds_fit_supervised.skip(3), ds_fit_supervised.skip(20))),
            data.mixup_datasets((ds_fit_supervised.skip(15), ds_fit_unsupervised.skip(200))),
            data.mixup_datasets((ds_fit_unsupervised.skip(350), ds_fit_supervised.skip(7))),
            data.mixup_datasets((ds_fit_unsupervised.skip(100), ds_fit_unsupervised.skip(500))),
        ),
        (
            config['n_supervised'],
            config['n_unsupervised'],
            config['n_mixup_supervised'],
            config['n_mixup_semisupervised'],
            config['n_mixup_semisupervised_reverse'],
            config['n_mixup_unsupervised']
        )
    )

    # it = iter(ds_fit)
    # import matplotlib.pyplot as plt
    # # %%
    # item = next(it)
    # plt.bar(np.arange(10), item[1])
    # plt.show()
    # plt.imshow(item[0][..., 0], cmap='gray')
    # plt.show()
    # # %%

    best_val_categorical_accuracy = 0
    def update_best(epoch, logs):
        global best_val_categorical_accuracy
        if logs['val_categorical_accuracy'] >= best_val_categorical_accuracy:
            best_val_categorical_accuracy = logs['val_categorical_accuracy']
            print(f' - best_val_categorical_accuracy: {best_val_categorical_accuracy:.4f}')

            test_metrics = model.evaluate(
                ds_test.batch(config['evaluation_batch_size'])
            )
            print(f' - test_categorical_accuracy: {test_metrics[1]:.4f}')


    os.makedirs('checkpoints')
    model.fit(
        ds_fit.batch(config['batch_size']),
        validation_data=ds_validate.batch(config['evaluation_batch_size']),
        epochs=config['max_epochs'],
        steps_per_epoch=config['steps_per_epoch'],
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir='tb',
                update_freq='batch',
                histogram_freq=0,
                write_graph=False,
                write_images=False
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/best_weights.h5',
                save_best_only=True,
                save_weights_only=True,
                monitor='val_categorical_accuracy',
                verbose=1
            ),
            # keras.callbacks.EarlyStopping(
            #     monitor='val_categorical_accuracy',
            #     mode='max',
            #     min_delta=1e-2,
            #     patience=20,
            #     verbose=1,
            #     restore_best_weights=True
            # ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=update_best
            ),
        ],
        verbose=1
    )


    print(f' - best_val_categorical_accuracy: {best_val_categorical_accuracy:.4f}')
