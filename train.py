from clearml import Task, OutputModel
import tensorflow as tf
import datetime
import cv2
import numpy as np
from pathlib import Path
import logging

from model import Model
from utils import create_images

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

parameter = {
    'batch_size': 32,
    'epochs': 5,
    'optimizer': {
        'learning_rate': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-07,
        'amsgrad': False,
    },
    'model': {
    }
}

keras = tf.keras

print("TensorFlow version:", tf.__version__)

task = Task.init(project_name='mnist-cnn', task_name='train mnist', output_uri='s3://dl-netzeband-test/')
reporter = task.get_logger()
logging.getLogger('clearml.Task').level=logging.WARNING
logging.getLogger('clearml.storage').level=logging.WARNING

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

test_images = x_test[0:10]

# upload artifacts
task.upload_artifact(name='dataset_train', artifact_object={
    'x': x_train,
    'y': y_train,
})
task.upload_artifact(name='dataset_test', artifact_object={
    'x': x_test,
    'y': y_test,
})

task.connect(parameter)
output_model = OutputModel(task=task)
output_model.update_design(config_dict=parameter['model'])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=parameter['optimizer']['learning_rate'],
    beta_1=parameter['optimizer']['beta_1'],
    beta_2=parameter['optimizer']['beta_2'],
    epsilon=parameter['optimizer']['epsilon'],
    amsgrad=parameter['optimizer']['amsgrad'],
    name='Adam',
)

model = Model()
model.compile(
    optimizer=optimizer,
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

def show_test_images(epoch, *args, **kwargs):
    test_labels = np.argmax(model.predict(test_images), axis=1)
    image = create_images(test_images, test_labels)
    reporter.report_image("image", "image uint8", iteration=epoch, image=image)
    cv2.imshow('image', image)
    cv2.waitKey(1)

image_callback = keras.callbacks.LambdaCallback(on_train_begin=show_test_images, on_epoch_end=show_test_images)

model.fit(
    x=x_train,
    y=y_train,
    epochs=parameter['epochs'],
    batch_size=parameter['batch_size'],
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, store_model_callback, image_callback]
)

task.close()