# References:
# 1. https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries
# 2. https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

cur_dir = os.getcwd();
train_dir = cur_dir + '/imgs/train'
train_bugs = 'bugs/*.jpg'
train_normal = 'normal/*.jpg'

print('** using train directory: ', train_dir)

# 训练集
dataset_dir = tf.keras.utils.get_file(train_dir, origin='')
dataset_dir = pathlib.Path(dataset_dir)

all_image_count = len(list(dataset_dir.glob('*/*.jpg')))
print('** all train images: ', all_image_count)

train_bugs_count = len(list(dataset_dir.glob(train_bugs)))
train_normal_count = len(list(dataset_dir.glob(train_normal)))
print('* train bugs: ', train_bugs_count)
print('* train normal: ', train_normal_count)

batch_size = 32
img_height = 120
img_width = 120
myseed = 427309
color = 'rgb'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_dir,
  # subset="training",
  seed=myseed,
  color_mode=color,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print('** classes: ', class_names)

for image_batch, labels_batch in train_ds:
  print('* image shape: ', image_batch.shape)
  print('* label shape: ', labels_batch.shape)
  break

# 验证集
val_dir = cur_dir + '/imgs/val'
print('** using validate directory: ', val_dir)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  # subset="validation",
  seed=myseed,
  color_mode=color,
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

# 数据转置，增加样本
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    # layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.Resizing(img_height, img_width),
#    layers.experimental.preprocessing.RandomWidth(0.1),
#    layers.experimental.preprocessing.RandomHeight(0.1)
  ]
)


# 模型
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
  layers.MaxPooling2D(),

  layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  
  layers.Flatten(),
  layers.Dense(128, activation='relu'),

  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 开始训练
epochs = 3
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#############################################################
# 测试集
test_dir = cur_dir + '/imgs/test/'
outputfile = open("output.csv", "w")

for testfile in os.listdir(test_dir):
	img = keras.preprocessing.image.load_img(
		test_dir + testfile, target_size=(img_height, img_width)
	)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	print("%s: %s, confidence:%.1f" % (testfile, class_names[np.argmax(score)], 100 * np.max(score)))
	outputfile.write("%s: %s\n" % (testfile, class_names[np.argmax(score)]))

outputfile.close()

model.save('saved_model.bin')
