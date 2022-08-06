# -*- coding: utf-8 -*-

"""
Created on Wed Jul 27 11:54:43 2022

@author: Admin
"""

from keras.preprocessing.image import ImageDataGenerator
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)    
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1)
x_train = train_datagen.flow_from_directory(batch_size=16,
                                                 directory='E:\\IBMProj\\Dataset Plant Disease\\veg-dataset\\veg-dataset\\train_set',
                                                 shuffle=True,
                                                 target_size=(128, 128), 
                                                 subset="training",
                                                 class_mode='categorical')
x_test = test_datagen.flow_from_directory(batch_size=16,
                                                 directory='E:\\IBMProj\\Dataset Plant Disease\\veg-dataset\\veg-dataset\\test_set',
                                                 shuffle=True,
                                                 target_size=(128, 128), 
                                                 subset="validation",
                                                 class_mode='categorical')
from tensorflow.python.keras.models import Sequential
from tensorflow import keras as ktf
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D,Flatten

model=ktf.models.Sequential()
model.add(Convolution2D(16,(3,3), input_shape=(128,128,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,(3,3), input_shape=(128,128,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(300,activation="relu"))
#model.add(Dense(200,activation="relu"))
model.add(Dense(150,activation="relu"))
model.add(Dense(75,activation="relu"))
model.add(Dense(9,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit_generator(x_train,steps_per_epoch=168,epochs=15,validation_data=x_test,validation_steps=54)
model.save("vegetable.h5")


