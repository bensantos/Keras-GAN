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

models = os.listdir(r'/home/ben/gans_git/Keras-GAN/monet/models/')
for i in range(len(models)):
    # load the image
    image_src = load_image(r'/home/ben/gans_git/auto_encoder_experiments/cocodataset/unlabeled2017/000000280262.jpg')
    # load the model
    cust = {'InstanceNormalization': InstanceNormalization}
    model_BtoA = load_model(r'/home/ben/gans_git/Keras-GAN/monet/models/{}'.format(models[i]), cust)
    # translate image
    image_tar = model_BtoA.predict(image_src)
    # scale from [-1,1] to [0,1]
    image_tar = (image_tar + 1) / 2.0
    # plot the translated image
    fig = pyplot.figure()
    pyplot.imshow(image_tar[0])
    pyplot.savefig("{}.jpg".format(models[i]))
