import numpy as np
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import matplotlib.image as mpimg

def get_image_paths(folder):
    paths = []
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            paths.append(os.path.abspath(os.path.join(dirpath, f)))

    #paths = os.listdir(os.path.abspath(folder))#os.walk(os.path.abspath(folder))
    paths_length = len(paths)
    halved_paths = []
    for i in range(int(paths_length/4)):
        halved_paths.append(paths[i])
    #return halved_paths
    return paths

def create_dataset(paths, IMG_HEIGHT, IMG_WIDTH, batch_size):
    img_data_array=[]
    #class_name=[]
    idx_list = np.random.randint(0, len(paths), batch_size)
    for index in idx_list:
        image_path = paths[index]
        image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
        image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image=np.array(image)
        image = image.astype('float32')
        img_data_array.append(image)
        #class_name.append(dir1)
    img_data_array = np.array(img_data_array)
    return img_data_array

#paths = get_image_paths(r"D:\Kaspar\unlabeled2017\unlabeled2017")
#images = create_dataset(paths, 256, 256, 128)
#print(images.shape)
