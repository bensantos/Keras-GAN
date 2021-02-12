# example of using saved cyclegan models for image translation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
 

# load an image to the preferred size
def load_image(filename, size=(256,256)):
    # load and resize the image
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # transform in a sample
    pixels = expand_dims(pixels, 0)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    return pixels

# load the image
image_src = load_image(r'D:\Kaspar\monet_dataset\downloads\extracted\monet2photo\trainB/2014-04-15 09_34_13.jpg')
# load the model
cust = {'InstanceNormalization': InstanceNormalization}
model_BtoA = load_model(r'D:\Kaspar\images\monet\g_model_BtoA_119300.h5', cust)
print("start")
# translate image
image_tar = model_BtoA.predict(image_src)
# scale from [-1,1] to [0,1]
image_tar = (image_tar + 1) / 2.0
# plot the translated image
pyplot.imshow(image_tar[0])
pyplot.show()