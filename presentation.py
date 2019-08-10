
# [ ] Background, Aiwizo, damages on railroads
# [ ] Research, may have varying degrees of success
# [ ] Why interesting? Use cases, when labelling is prohibitive
#   Expensive
#   Difficult
# [ ] When might this happen?
#   Imbalanced datasets
#   Online learning?
# [ ] Limitations
#   How to mixup non-categorical?
# [x] show problem
# [x] show augmentation
# [x] show mixup
# [x] show sharpening function
# [x] show results


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,10)

import problem


default_train_size = 0.0002

avg_images_per_class = list(map(
    lambda x: np.mean(problem.training_count(x)[1]),
    default_train_size*2**np.linspace(0, 3, num=4)
))

avg_images_per_class
# %%

import data

image_train, label_train = problem.get_data(
    problem.TRAIN,
    train_size=default_train_size
)

ds_train = data.get_standard_ds(image_train, label_train)

ds_train
# %%

plt.imshow(next(iter(ds_train))[0][:,:,0], cmap='gray')
plt.show()
# %%

ds_train_augmented = ds_train.map(lambda image, label: (
    data.augment(image), label
))
plt.imshow(next(iter(ds_train_augmented))[0][:,:,0], cmap='gray')
plt.show()
# %%

# Mixup with defined weights

image1 = next(iter(ds_train_augmented))[0][:,:,0]
image2 = next(iter(ds_train_augmented.skip(2)))[0][:,:,0]

q = 0.8
mixup_image = image1*q + image2*(1 - q)


plt.imshow(mixup_image, cmap='gray')
plt.show()
# %%

### Mixup weights
# Favoring the first image

import tensorflow as tf
import tensorflow_probability as tfp

q = tfp.distributions.Beta(0.5, 0.5).sample((10000,))
q = tf.maximum(q, 1 - q)

plt.hist(q.numpy(), bins=30)
plt.show()
# %%

### Mixup
# Sounds also works nicely

image1 = next(iter(ds_train_augmented))[0][:,:,0]
image2 = next(iter(ds_train_augmented.skip(2)))[0][:,:,0]

q = tfp.distributions.Beta(0.5, 0.5).sample((1,))
q = tf.maximum(q, 1 - q)
mixup_image = image1*q + image2*(1 - q)

plt.imshow(mixup_image, cmap='gray')
plt.show()
# %%

### Visualize sharpening

p = np.random.dirichlet(np.ones(10)*0.8 + np.eye(10)[3]*2)

sharpen_exponent = 6
sharpened_p = p**sharpen_exponent / np.sum(p**sharpen_exponent)

plt.bar(np.arange(10), p, alpha=0.5, label='p')
plt.bar(np.arange(10), sharpened_p, alpha=0.5, label='sharpened_p')
plt.legend()
plt.show()
# %%

### Creating the unsupervised dataset
# Label guessing, averaging, and sharpening

if False:
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
        .batch(config['n_predictions'])
        .map(lambda predictions: tf.reduce_mean(predictions, axis=0))
        .map(lambda prediction: data.sharpen(prediction, config['sharpen_exponent']))
    )

    ds_fit_unsupervised = (
        tf.data.Dataset.zip((
            ds_validate_shuffled.map(lambda image, _: image),
            ds_predictions_shuffled
        ))
        .map(lambda image, label: (
            data.augment(image), label
        ))
    )


# %%

### Mixmatch / Mixup / Other
# Asymmetric weights

if False:
    ds_fit = data.merge_datasets(
        (
            ds_fit_supervised,
            ds_fit_unsupervised,
            data.mixup_datasets((ds_fit_supervised, ds_fit_supervised)),
            data.mixup_datasets((ds_fit_supervised, ds_fit_unsupervised)),
            data.mixup_datasets((ds_fit_unsupervised, ds_fit_supervised)),
            data.mixup_datasets((ds_fit_unsupervised, ds_fit_unsupervised)),
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

# %%

import pandas as pd

# !guild compare --csv > results.csv
results = pd.read_csv('results.csv')

results = results[results['label'] == 'investigate_problem_size']
results = results[~((results['n_supervised'] == 1) & (results['n_unsupervised'] == 1))]

# results = results[results['seed'].isin([421, 377, 789])]
# results = results[results['learning_rate'] == 0.004]
# results = results[
#     ((results['n_supervised'] == 1) & (results['n_mixup_semisupervised'] == 0)) |
#     ((results['n_mixup_supervised'] == 1) & (results['n_mixup_semisupervised'] == 0)) |
#     (results['n_mixup_semisupervised'] == 1)
# ]

results['type'] = ''
results.loc[(results['n_supervised'] == 1) & (results['n_unsupervised'] == 0), 'type'] = 'supervised'
# results.loc[(results['n_supervised'] == 1) & (results['n_unsupervised'] == 1), 'type'] = 'semisupervised'
results.loc[results['n_mixup_supervised'] == 1, 'type'] = 'mixup_supervised'
results.loc[results['n_mixup_semisupervised'] == 1, 'type'] = 'mixup_semisupervised'

results['avg_images_per_class'] = results['problem_train_size']/0.0002*1.2

results['log_avg_images_per_class'] = np.log(results['problem_train_size']/0.0002*1.2)

results['improvement'] = results['test_categorical_accuracy']

table = pd.pivot_table(results, index='avg_images_per_class', columns='type', values='test_categorical_accuracy')

table
# %%

ax = table.plot()
ax.set_ylabel('test_categorical_accuracy')
plt.show()
# %%

# ((table['mixup_semisupervised'] - table['mixup_supervised'])/(1 - table['mixup_supervised'])).plot()
# plt.title('relative improvement')
# plt.show()
# %%

# Imbalanced datasets
# Unable to stratify with labels - weighted sampling (i.e. by CE)
