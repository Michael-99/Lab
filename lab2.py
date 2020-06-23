from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/My Drive/"

!ls "/content/drive/My Drive/xray/"

import argparse
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from datetime import datetime

LOG_DIR = '/content/drive/My Drive/lab2'
SHUFFLE_BUFFER = 10
BATCH_SIZE = 5
NUM_CLASSES = 50
PARALLEL_CALLS=4
RESIZE_TO = 224
TRAINSET_SIZE = 5216
VALSET_SIZE=624

def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    example = tf.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize_images(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image'], example['image/class/label']


def normalize(image, label):
    return tf.image.per_image_standardization(image), label

def resize(image, label):
    return tf.image.resize_images(image, tf.constant([RESIZE_TO, RESIZE_TO])), label

def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .map(resize)\
        .map(normalize)\
        .shuffle(buffer_size=5 * batch_size)\
        .repeat()\
        .batch(batch_size)\
        .prefetch(2 * batch_size)

def build_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(224,224,3)),
        # 6
        #tf.keras.layers.Conv2D(filters=8, kernel_size=7),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=7),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=4),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=4),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        # 5
        tf.keras.layers.Conv2D(filters=8, kernel_size=5),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=5),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=4),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=4),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        tf.keras.layers.MaxPool2D(),
        # 4 
      
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=4),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=5),
        #tf.keras.layers.MaxPool2D(),
        # 3
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=4),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=5),
        #tf.keras.layers.MaxPool2D(),
     
        # 2
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #1
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        #tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)
    ])

class Validation(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, validation_files, batch_size):
        self.log_dir = log_dir
        self.validation_files = validation_files
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        print('The average loss for epoch {} is {:7.2f} '.format(
            epoch, logs['loss']
        ))

        validation_dataset = create_dataset(self.validation_files, self.batch_size)
        validation_images, validation_labels = validation_dataset.make_one_shot_iterator().get_next()
        validation_labels = tf.one_hot(validation_labels, NUM_CLASSES)

        result = self.model.evaluate(
            validation_images,
            validation_labels,
            steps=int(np.ceil(VALSET_SIZE / float(BATCH_SIZE)))
        )
        callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, update_freq='epoch', batch_size=self.batch_size)

        callback.set_model(self.model)
        callback.on_epoch_end(epoch, {
            'val_' + self.model.metrics_names[i]: v for i, v in enumerate(result)
        })

train_path = '/content/drive/My Drive/xray/train*'
test_path = '/content/drive/My Drive/xray/test*'
train_dataset = create_dataset(glob.glob(train_path), BATCH_SIZE)
train_images, train_labels = train_dataset.make_one_shot_iterator().get_next()
train_labels = tf.one_hot(train_labels, NUM_CLASSES)

model = build_model()
model.compile(
        optimizer=keras.optimizers.sgd(lr=0.0001, momentum=0.9),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy],
        target_tensors=[train_labels]
)
model.summary()

log_dir='{}/xray-{}'.format(LOG_DIR, datetime.now())

model.fit(
    (train_images, train_labels),
     epochs=60,
     steps_per_epoch=int(np.ceil(TRAINSET_SIZE / float(BATCH_SIZE))),
     callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            Validation(log_dir, validation_files=glob.glob(test_path), batch_size=BATCH_SIZE)
        ]
)
