import imageio
import numpy as np

from PIL import Image, ImageOps
import torch
import os
import time
import cv2

from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial

import matplotlib.pyplot as plt
def resize_and_pad(image:np.array, scale_size = 640,scale_smaller_images=True,*args,**kwargs):

    #old_size = image.shape  # old_size[0] is in (width, height) format
    #ratio = float(scale_size) / max(old_size)
    #new_size = tuple([int(x * ratio) for x in old_size])
    #new_im = np.zeros((scale_size, scale_size,3),np.float32)
    #mask = np.zeros((scale_size, scale_size, 3), np.float32)
    #x_min = min(old_size[0],scale_size)
    #y_min = min(old_size[1],scale_size)

    #new_im[:x_min,:y_min,:] = image[:x_min,:y_min,:]
    #mask[:x_min,:y_min,:] = 1
    #mask[:,:] = 1
    mask = np.ones((scale_size, scale_size, 3), np.float32)
    new_im = cv2.resize(image,(scale_size,scale_size))
    new_im = new_im /255

    return new_im, mask

def resize_and_pad_old(image:Image.Image, scale_size = 640,scale_smaller_images=True,*args,**kwargs):

    old_size = image.size  # old_size[0] is in (width, height) format


    ratio = float(scale_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)
    if(not scale_smaller_images):
        if(ratio < 1):
            image = image.resize(new_size, Image.ANTIALIAS)
    else:
        if(ratio != 1):
            image = image.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (scale_size, scale_size))
    new_im.paste(image)
    #new_im.paste(image, ((scale_size - new_size[0]) // 2,
    #                  (scale_size - new_size[1]) // 2))

    return new_im



def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]


from itertools import product
class ImageLoader():

    def __init__(self,thread_count):
        self.thread_count = thread_count
        self.pool = Pool(processes=self.thread_count)

    def load_images_parallel(self,paths,  to_cuda = True,*args,**kwargs):


        images = []
        masks = []
        chunked_paths = chunk(paths,self.thread_count)
        out = self.pool.map(partial(load_images_from_paths_new, **kwargs),chunked_paths)
        for i in out:
            images.extend(i[0])
            masks.extend(i[1])

        images = np.array(images)
        images = torch.from_numpy(images)
        masks = np.array(masks)
        masks = torch.from_numpy(masks)
        if (to_cuda):
            images = images.to("cuda:0")
            masks = masks.to("cuda:0")
        images = torch.transpose(images, 1, 3).float()
        masks = torch.transpose(masks, 1, 3)
        return images,masks

    def crop_images(self,imgs: torch.Tensor):
        device = imgs.device  # imgs.get_device()
        imgs = imgs.cpu().detach().numpy()
        output_images = []
        ratio = np.random.uniform(0.3, 0.7, (len(imgs)))
        which_half = np.random.uniform(0.0, 1, (len(imgs)))
        for ind, img in enumerate(imgs):
            non_zero_inds = np.where(img > 0)
            x_max = np.max(non_zero_inds[1])
            y_max = np.max(non_zero_inds[2])
            temp_img = img
            if (x_max > y_max):
                if (which_half[ind] > 0.5):
                    temp_img[:,:int(x_max * ratio[ind]), :] = 0
                else:
                    temp_img[:,int(x_max * ratio[ind]):, :] = 0
            else:
                if (which_half[ind] > 0.5):
                    temp_img[:,:, :int(y_max * ratio[ind])] = 0
                else:
                    temp_img[:,:, int(y_max * ratio[ind]):] = 0
            output_images.append(temp_img)
        output_images = torch.from_numpy(np.array(output_images)).float().to(device)
        return output_images


    def circle_crop_images(self,imgs: torch.Tensor,num_circles=10,rad_min=10,rad_max = 30):
        device = imgs.device  # imgs.get_device()
        imgs = imgs.cpu().detach().numpy()
        output_images = []
        for ind, img in enumerate(imgs):
            non_zero_inds = np.where(img > 0)
            x_max = np.max(non_zero_inds[1])
            y_max = np.max(non_zero_inds[2])
            temp_img = img

            for c in range(num_circles):
                center = np.array([np.random.randint(0,x_max), np.random.randint(0, y_max)])
                rad = np.random.uniform(rad_min,rad_max)
                circle_inds = self.get_circle_cords(center,rad,np.shape(imgs)[-1])

                #plt.figure()
                #plt.scatter(circle_inds[:,0],circle_inds[:,1])
                #plt.show()

                #for p in circle_inds:
                    #temp_img[:,p[0],p[1]] = 0
                temp_img[:, circle_inds[:,0],circle_inds[:,1]] = -1

            output_images.append(temp_img)
        output_images = torch.from_numpy(np.array(output_images)).float().to(device)
        return output_images

    def box_crop_images(self,imgs: torch.Tensor,num_boxes=10,size_min=10,size_max = 50):
        device = imgs.device  # imgs.get_device()
        imgs = imgs.cpu().detach().numpy().astype(np.float32)
        output_images = []
        for ind, img in enumerate(imgs):
            non_zero_inds = np.where(img > 0)
            x_max = np.max(non_zero_inds[1])
            y_max = np.max(non_zero_inds[2])
            temp_img = img

            for c in range(num_boxes):
                upper_left = np.array([np.random.randint(0,x_max), np.random.randint(0, y_max)])
                rad = np.random.random_integers(size_min,size_max,2)
                lower_right = upper_left+rad
                temp_img[:, upper_left[0]:lower_right[0],upper_left[1]:lower_right[1]] = -1

            output_images.append(temp_img.astype(np.float32))
        output_images = torch.from_numpy(np.array(output_images).astype(np.float32)).float().to(device)
        return output_images

    def get_circle_cords(self,center,rad,img_size=640):
        center = np.array(center)
        img = [x for x in range(img_size)]
        img = product(img,img)
        points = []
        for p in img:
            dist = np.linalg.norm(center-p)
            if(dist <= rad):
                points.append(p)
        points = np.array(points)
        return points

    def get_square_cords(self,upper_right,min_size,max_size,img_size=640):
        center = np.array(center)
        img = [x for x in range(img_size)]
        img = product(img,img)
        points = []
        for p in img:
            dist = np.linalg.norm(center-p)
            if(dist <= rad):
                points.append(p)
        points = np.array(points)
        return points





def load_images_from_paths_new(paths,*args,**kwargs):
    images = []
    masks = []
    for i, p in enumerate(paths):
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image,mask = resize_and_pad(image, *args, **kwargs)
        images.append(image)
        masks.append(mask)

    return images,masks

def load_images_from_paths(paths,  to_cuda = True,*args,**kwargs):

    images = []
    masks = []
    for i,p in enumerate(paths):

        image = cv2.imread(p)
        image, mask = resize_and_pad(image, *args, **kwargs)
        images.append(image)
        masks.append(mask)
    images = np.array(images)
    images = torch.from_numpy(images)
    masks = np.array(masks)
    masks = torch.from_numpy(masks)
    if(to_cuda):
        images = images.to("cuda:0")
        masks = masks.to("cuda:0")
    images = torch.transpose(images,1,3)
    masks = torch.transpose(masks, 1, 3)

    return images,masks


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
    return halved_paths
    #return paths