# refer: https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

cur_dir = os.getcwd();
train_dir = cur_dir + '/meipian-catdog-data/train'
train_dogs = 'dog.jpg/*.jpg'
train_cats = 'cat.jpg/*.jpg'


dataset_dir = tf.keras.utils.get_file(train_dir, origin='')
dataset_dir = pathlib.Path(dataset_dir)

all_image_count = len(list(dataset_dir.glob('*/*.jpg')))
print('** all train images: ', all_image_count)

train_dogs_count = len(list(dataset_dir.glob(train_dogs)))
train_cats_count = len(list(dataset_dir.glob(train_cats)))
print('* train dogs: ', train_dogs_count)
print('* train cats: ', train_cats_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print('** classes: ', class_names)

for image_batch, labels_batch in train_ds:
  print('* image shape: ', image_batch.shape)
  print('* label shape: ', labels_batch.shape)
  break

val_dir = cur_dir + '/meipian-catdog-data/val'
# val_dogs = 'dog.*.jpg'
# val_cats = 'cat.*.jpg'

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# # Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)