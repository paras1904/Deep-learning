import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils

mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = '/home/paras/Data/Mobilenet-samples/'
    img = image.load_img(img_path+file,target_size = (224,224))
    img_array = image.img_to_array(img)
    img_array_expand_dims = np.expand_dims(img_array,axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expand_dims)

from IPython.display import Image
Image(filename = '/home/paras/Data/Mobilenet-samples/1.PNG',width = 300,height = 200)

preprocessed_image = prepare_image('1.PNG')
predictions=mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)