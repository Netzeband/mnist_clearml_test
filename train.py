from clearml import Task
import tensorflow as tf
import datetime
import cv2
import numpy as np
from pathlib import Path

from model import Model
from utils import create_images

keras = tf.keras

print("TensorFlow version:", tf.__version__)

task = Task.init(project_name='mnist-cnn', task_name='train mnist')

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

test_images = x_test[0:10]


model = Model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

log_dir = Path("logs/fit") / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_dir = Path("model_files/model")
model_dir.parent.mkdir(exist_ok=True)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
store_model_callback = keras.callbacks.ModelCheckpoint(
    filepath=model_dir,
    save_weights_only=False,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
)

def show_test_images(*args, **kwargs):
    test_labels = np.argmax(model.predict(test_images), axis=1)
    image = create_images(test_images, test_labels)
    print(image.shape)
    cv2.imshow('image', image)
    cv2.waitKey(1)

image_callback = keras.callbacks.LambdaCallback(on_train_begin=show_test_images, on_epoch_end=show_test_images)

model.fit(
    x=x_train,
    y=y_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, store_model_callback, image_callback]
)

cv2.waitKey(0)