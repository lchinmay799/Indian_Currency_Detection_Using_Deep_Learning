import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras

img=cv2.cvtColor(cv2.imread("/home/chinmay/Desktop/projects/VI_Semester/MDP/Datasets/datasets/ten/10__32.jpg"), cv2.COLOR_BGR2RGB)

img=img_to_array(img).astype('int64')
print("Shape of a Sample Image : ",img.shape)
print("======="*12, end="\n\n\n")

plt.imshow(img)
plt.show()

height,width=200,200

resized_image =img_to_array(img)
resized_image=cv2.resize(resized_image, (height, width),
               interpolation = cv2.INTER_NEAREST).astype('int64')
print("Highest Value of the Pixels in the resized image : ",np.max(np.array(resized_image).reshape(1,-1)))
print("======="*12, end="\n\n\n")

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

print("======="*12, end="\n\n\n")
print("Size of Daataset : X and Y : ",len(x),len(y))
print("======="*12, end="\n\n\n")

import random
temp = list(zip(x, y))
random.shuffle(temp)
x, y = zip(*temp)

x=list(x)
y=list(y)
print("Size of Daataset : X and Y (After Shuffeling) : ",len(x),len(y))
print("======="*12, end="\n\n\n")

plt.imshow(x[1])
plt.show()
print(y[1])

encoder=LabelEncoder()
y=encoder.fit_transform(y)

print("After Encoding, Class Names : ",set(y))
print("Actual Classnames : ",encoder.classes_)
print("======="*12, end="\n\n\n")

x=np.array(x)
y=np.array(y)

print("Shaape of x and y : ",x.shape,y.shape)
print("======="*12, end="\n\n\n")
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
#ratio=0.1
#ratio=int(ratio*x.shape[0])
#x_train,x_test,y_train,y_test=x[ratio:],x[:ratio],y[ratio:],y[:ratio]

print("Train Dataset Shape : ",x_train.shape,y_train.shape,"\nTest Dataset Shape : ",x_test.shape,y_test.shape)
print("======="*12, end="\n\n\n")

x_valid,y_valid=x_train[:900],y_train[:900]
x_train,y_train=x_train[900:],y_train[900:]

print("======="*12, end="\n\n\n")
print("Train Dataset Shape : ",x_train.shape,y_train.shape,"\nTest Dataset Shape : ",x_test.shape,y_test.shape,"\nValidation Dataset Shape : ",x_valid.shape,y_valid.shape)

inputs=keras.layers.Input(shape=[height,width,3])
hidden1=keras.layers.Conv2D(32,(3,3),strides=(2,2),activation='relu')(inputs)
hidden1_pool=keras.layers.MaxPooling2D(2,2)(hidden1)
hidden2=keras.layers.Conv2D(32,(1,1),strides=(1,1),activation='relu')(hidden1_pool)
concat1=keras.layers.Concatenate()([hidden1_pool,hidden2])
hidden3=keras.layers.Conv2D(64,(1,1),strides=(1,1),activation='relu')(concat1)
hidden4=keras.layers.Conv2D(64,(1,1),strides=(1,1),activation='relu')(hidden3)
concat2=keras.layers.Concatenate()([hidden3,hidden4])
concat3=keras.layers.Concatenate()([concat1,concat2])
concat3_pool=keras.layers.MaxPooling2D(2,2)(concat3)
hidden5=keras.layers.Conv2D(128,(2,2),strides=(1,1),activation='relu')(concat3_pool)
hidden5_pool=keras.layers.MaxPooling2D(2,2)(hidden5)
hidden6=keras.layers.Conv2D(128,(1,1),strides=(1,1),activation='relu')(hidden5_pool)
concat4=keras.layers.Concatenate()([hidden5_pool,hidden6])
hidden7=keras.layers.Conv2D(256,(1,1),strides=(1,1),activation='relu')(concat4)
hidden8=keras.layers.Conv2D(256,(1,1),strides=(1,1),activation='relu')(hidden7)
concat5=keras.layers.Concatenate()([hidden7,hidden8])
concat6=keras.layers.Concatenate()([concat4,concat5])
concat6_pool=keras.layers.MaxPooling2D(2,2)(concat6)
hidden9=keras.layers.Conv2D(512,(1,1),strides=(1,1),activation='relu')(concat6_pool)
hidden10=keras.layers.Conv2D(512,(1,1),strides=(1,1),activation='relu')(hidden9)
concat7=keras.layers.Concatenate()([hidden9,hidden10])
concat8=keras.layers.Concatenate()([concat6_pool,concat7])
concat8_pool=keras.layers.MaxPooling2D(2,2)(concat8)
flat=keras.layers.Flatten()(concat8_pool)
nn1=keras.layers.Dense(256,activation='relu')(flat)
nn2=keras.layers.Dense(512,activation='relu')(nn1)
nn3=keras.layers.Dense(256,activation='relu')(nn2)
nn4=keras.layers.Dense(128,activation='relu')(nn3)
nn5=keras.layers.Dense(64,activation='relu')(nn4)
nn6=keras.layers.Dense(28,activation='relu')(nn5)
outputs=keras.layers.Dense(7,activation='softmax')(nn6)
model=keras.models.Model(inputs=[inputs],outputs=[outputs])

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(learning_rate=1e-4),metrics=['accuracy'])

early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1)
model_checkpoint=keras.callbacks.ModelCheckpoint('Indian_Currency_Detection.h5',save_best_only=True,save_freq='epoch',mode='auto')

#n_folds=10
#validation_size=int(x_train.shape[0]/n_folds)

#for i in range(n_folds):
#	print("Training on Fold ",i+1," : ",end="\n\n\n\n")
#	start=i*validation_size
#	end=start+validation_size
#	x_valid,x_train=x_train[start:end],x_train[end:]
#	y_valid,y_train=y_train[start:end],y_train[end:]

model.fit(x_train,y_train,epochs=8,batch_size=1,validation_data=(x_valid,y_valid),callbacks=[early_stopping,model_checkpoint])
model=keras.models.load_model('Indian_Currency_Detection.h5')
#	print("======="*12, end="\n\n\n")

model.evaluate(x_test,y_test)

y_pred=np.argmax(model.predict(x_test),axis=-1)

print("Predicted Output : ",y_pred)
print("Actual Output : ",y_test)
print("======="*12, end="\n\n\n")

print("Accuracy of Neural Network: ",100*len([x for x in y_pred==y_test if x==True])/len(y_pred))
print("======="*12, end="\n\n\n")

model.save('Indian_Currency_Detection.h5')
