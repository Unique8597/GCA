from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np
import tensorflow as tf

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, 0)
    return img_array
