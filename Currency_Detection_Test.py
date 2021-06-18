import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

height,width=200,200

img=cv2.cvtColor(cv2.imread("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/ten/10__32.jpg"), cv2.COLOR_BGR2RGB)

img=img_to_array(img).astype('int64')
print("Shape of a Sample Image : ",img.shape)

plt.imshow(img)
plt.show()

height,width=200,200

resized_image =img_to_array(img)
resized_image=cv2.resize(resized_image, (height, width),
               interpolation = cv2.INTER_NEAREST).astype('int64')
print("Highest Value of the Pixels in the resized image : ",np.max(np.array(resized_image).reshape(1,-1)))

plt.imshow(resized_image)
plt.show()

resized_image.shape

print(resized_image)

def load_images_from_folder(folder,y,a):
    images = []
    for filename in os.listdir(folder):
        img= cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(folder,filename)), cv2.COLOR_BGR2RGB),(height,width), 
                           interpolation=cv2.INTER_NEAREST)
        if img is not None:
            y.append(a)
            images.append(img)
    return images,y


x,y=[],[]
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/ten",y,10)
x.extend(img)
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/twenty",y,20)
x.extend(img)
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/two_thousand",y,50)
x.extend(img)
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/hundred",y,100)
x.extend(img)
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/two_hundred",y,200)
x.extend(img)
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/five_hundred",y,500)
x.extend(img)
img,y=load_images_from_folder("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/five_hundred",y,2000)
x.extend(img)

print("Size of Daataset : X and Y : ",len(x),len(y))

import random
temp = list(zip(x, y))
random.shuffle(temp)
x, y = zip(*temp)

x=list(x)
y=list(y)
print("Size of Daataset : X and Y (After Shuffeling) : ",len(x),len(y))

plt.imshow(x[1])
plt.show()
print(y[1])

encoder=LabelEncoder()
y=encoder.fit_transform(y)

print("After Encoding, Class Names : ",set(y))
print("Actual Classnames : ",encoder.classes_)

x=np.array(x)
y=np.array(y)

print("Shape of x and y : ",x.shape,y.shape)

model=keras.models.load_model('Indian_Currency_Detection.h5')

model.summary()

model.evaluate(x,y)

y_pred=np.argmax(model.predict(x),axis=-1)

print("Predicted Output : ",y_pred)
print("Actual Output : ",y)

print("Accuracy of Neural Network on the Entore Dataset: ",100*len([x for x in y_pred==y if x==True])/len(y_pred))

model.save('Indian_Currency_Detection.h5')
