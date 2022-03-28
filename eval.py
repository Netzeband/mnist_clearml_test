import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

from utils import create_images

keras = tf.keras

print("TensorFlow version:", tf.__version__)

model_dir = Path("model_files/model")
mnist = keras.datasets.mnist

_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255
x_test = x_test[..., tf.newaxis].astype("float32")


model = keras.models.load_model(model_dir)
prediction = np.argmax(model.predict_on_batch(x_test), axis=1)

m = keras.metrics.Accuracy()
m.update_state(y_test, prediction)
print(f"Accuracy: {m.result().numpy()*100:.2f}%")

cv2.imshow('image', create_images(x_test[0:20], prediction[:20]))
cv2.waitKey(0)




