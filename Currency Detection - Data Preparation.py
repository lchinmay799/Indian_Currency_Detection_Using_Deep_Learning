#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image as im
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array


# In[2]:


img=load_img("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/ten/10__333.jpg")


# In[3]:


image_array=img_to_array(img)
print(type(image_array),image_array.shape)


# In[4]:


plt.imshow(img)


# In[5]:


height,width=200,200


# In[6]:


resized_image =img_to_array(img)
resized_image=cv2.resize(resized_image, (height, width),
               interpolation = cv2.INTER_NEAREST).astype('int64')
print(np.max(np.array(resized_image).reshape(1,-1)))


# In[7]:


plt.imshow(resized_image)


# In[8]:


resized_image.shape


# In[9]:


print(resized_image)


# In[10]:


def generate_image(image,folder):
    i=0
    datagen = ImageDataGenerator(rotation_range=90,width_shift_range=0.2,height_shift_range=0.2,
                                 zoom_range=0.25,shear_range=0.25,fill_mode="nearest")
    for batch in datagen.flow(image,batch_size=1,save_to_dir=folder,save_format='jpg'):
        i+=1
        if i>3:
            break


# In[11]:


def load_images_from_folder(folder,target_folder):
    for filename in os.listdir(folder):
        img= img_to_array(load_img(os.path.join(folder,filename)))
        img=img.reshape((1,)+img.shape)
        if img is not None:
            generate_image(img,target_folder)


# In[12]:


load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/ten","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/ten")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/twenty","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/twenty")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/fifty","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/fifty")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/hundred","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/hundred")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/two_thousand","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/two_hundred")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/five_hundred","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/five_hundred")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/two_hundred","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/two_thousand")
load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets_Copy/datasets/Background","/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/Background")

