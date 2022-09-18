import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple

def load_image(image_path: str) -> np.array:
    img = Image.open(image_path)
    img = np.asarray(img)
    return img

def process_image(image: np.array, 
                  image_size: int = 224) -> np.array:
    '''A simple function to process a given image
       before feed the model with it'''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

def predict(img: np.array, 
            model: tf.keras.models, 
            top_k: int = 5) -> Tuple:
    '''Return the `top_k` most probable classes of the given
       image using a pretraining model'''
    img = np.expand_dims(img, axis = 0)
    preds   = model.predict(img)[0]
    top_values_idx = np.argpartition(preds, -top_k)[-top_k:]
    top_values_idx = top_values_idx[np.argsort(preds[top_values_idx])][::-1]
    probs   = preds[top_values_idx]
    # add one so the first class will be 1 instead of 0
    classes = top_values_idx + 1 
    classes = [str(c) for c in classes]
    return probs, classes
