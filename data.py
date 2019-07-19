import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
import os
import albumentations
from functools import reduce
import argparse

import problem


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

albumentations_augmentor = albumentations.Compose([
    # albumentations.IAAPerspective(always_apply=True),
    # albumentations.IAAAffine(always_apply=True),
    albumentations.OneOf([
        albumentations.OpticalDistortion(p=0.3, border_mode=0),
        albumentations.GridDistortion(num_steps=3, distort_limit=0.3, border_mode=0),
        albumentations.ElasticTransform(alpha=10, sigma=5, alpha_affine=5, border_mode=0),
    ], p=0.8),
    # albumentations.OneOf([
        # albumentations.IAAAdditiveGaussianNoise(),
        albumentations.GaussNoise(var_limit=(0.01, 0.05)),
    # ], p=0.4),
    albumentations.OneOf([
        albumentations.MotionBlur(p=0.2, always_apply=True),
        albumentations.MedianBlur(blur_limit=3, p=0.1),
        albumentations.Blur(blur_limit=1, p=0.1),
    ], p=0.4),
    albumentations.OneOf([
        albumentations.IAASharpen(),
        albumentations.IAAEmboss(),
        albumentations.RandomBrightnessContrast(brightness_limit=0.9, contrast_limit=0.9),
    ], p=0.5),
    # albumentations.RandomGamma(),
    # albumentations.RandomScale(),
    # albumentations.RandomSnow(),
    # albumentations.RandomRain(),
    # albumentations.RandomFog(),
    # albumentations.CoarseDropout(),
])

def random_transform(image):
    image = image_augmentor.random_transform(image)
    return image
    # result = albumentations_augmentor(image=image)# * 255.0)
    # return result['image'] #/ 255.0


# images = problem.get_data(problem.VALIDATE)[0]
# import matplotlib.pyplot as plt
# image = random_transform(preprocess_image(images[0]))[...,0]
# print(image.min(), image.max())
# plt.imshow(image, vmin=0.0, vmax=1.0, cmap='gray')
# %%
def augment(image):
    return tf.reshape(tf.numpy_function(
        func=random_transform,
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


def predict_batch(model, image):
    return tf.reshape(tf.py_function(
        func=model.predict_on_batch,
        inp=[image],
        Tout=tf.float32,
    ), (-1, 10,))


def merge_datasets(datasets, ns):
    return (
        tf.data.Dataset.zip(tuple(ds.batch(n) for ds, n in zip(datasets, ns) if n >= 1))
        .flat_map(lambda *batches: reduce(tf.data.Dataset.concatenate, [
            tf.data.Dataset.from_tensors(batch).unbatch()
            for batch in batches
        ]))
    )


def get_mixup_weights():
    # p = tfp.distributions.Beta(a, b).sample(shape)
    p = tfp.distributions.Beta(0.5, 0.5).sample((1,))
    p = tf.maximum(p, 1 - p)
    return tf.concat([p, (1 - p)], axis=0)


def mixup_items(items):
    weights = get_mixup_weights()
    weights /= tf.reduce_sum(weights)

    return tuple([
        tf.einsum('i...,i->...', tf.stack(variable, axis=0), weights)
        for variable in zip(*items)
    ])


def mixup_datasets(datasets):
    return (
        tf.data.Dataset.zip(datasets)
        .map(lambda *items: mixup_items(items))
    )
