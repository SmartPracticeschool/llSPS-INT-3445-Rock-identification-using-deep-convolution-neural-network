import keras
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator    
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
x_train=train_datagen.flow_from_directory('/home/student/Desktop/dataset/dataset/training_set',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=train_datagen.flow_from_directory('/home/student/Desktop/dataset/dataset/test_set',target_size=(64,64),batch_size=32,class_mode='categorical')
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy  
from tensorflow.keras.layers import Convolution2D 
from tensorflow.keras.layers import MaxPooling2D 
from tensorflow.keras.layers import Flatten 
model=Sequential() 
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=128,activation='relu',kernel_initializer='random_uniform'))
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='random_uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(x_train,steps_per_epoch=7,epochs=10,validation_data=x_test,validation_steps=3)
model.save('project3.h5')






