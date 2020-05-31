import numpy as np
import pandas as pd
from os import listdir
from PIL import Image as PImage

from PIL import Image
import os, os.path

#import cv
import os
import glob
import matplotlib.pyplot as plt
#
# img_dir = "C:/Users/Sana Kanwal/Downloads/Compressed/dog vs cat/dataset/training_set"  # Enter Directory of all images
# Categories=['cat','dog']
# data_path = os.path.join(img_dir, '*g')
# files = glob.glob(data_path)
# data = []
# for f1 in files:
#     img = cv.imread(f1)
#     data.append(img)
# print('Images loaded')
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()
''' First Conv and Max pool Hidden Layer'''
classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

'''Second  Conv and Max pool Hidden Layer'''
classifier.add(Conv2D(16,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

'''Flatten Layer'''
classifier.add(Flatten())

'''Fully Connected Layer'''
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

'''Compile'''
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print('Done')
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory("C:/Users/Sana Kanwal/Downloads/Compressed/dog vs cat/dataset/training_set",target_size=(64,64),batch_size=32,class_mode='binary')
testing_set=test_datagen.flow_from_directory("C:/Users/Sana Kanwal/Downloads/Compressed/dog vs cat/dataset/test_set",target_size=(64,64),batch_size=32,class_mode='binary')
print('upload')
from IPython.display import display
from PIL import Image
history=classifier.fit_generator(training_set,steps_per_epoch=7000,epochs=10,validation_data=testing_set,validation_steps=2000)
train_accu=classifier.evaluate_generator(training_set)
print('training accuracy',train_accu)
test_accu=classifier.evaluate_generator(testing_set)
print('Testing accurau',test_accu)
print('Model fitted')
# result={0:'cat',1:'dog'}
# from PIL import Image
# import numpy as np
# im=Image.open("C:/Users/Sana Kanwal/Desktop/dog.10.jpg")
# #im=im.resize(Image_Size)
# im=np.expand_dims(im,axis=0)
# im=np.array(im)
# im=im/255
# pred=classifier.predict_classes([im])[0]
# print(pred,result[pred])
#
# im=Image.open("C:/Users/Sana Kanwal/Desktop/cat1.jpg")
# #im=im.resize(Image_Size)
# im=np.expand_dims(im,axis=0)
# im=np.array(im)
# im=im/255
# pred=classifier.predict_classes([im])[0]
# print(pred,result[pred])
import numpy as np
from keras.preprocessing import image
test_img=image.load_img('C:/Users/Sana Kanwal/Downloads/Compressed/23_29dataset/single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
result=classifier.predict(test_img)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
    print(prediction)
else:
    prediction='cat'
    print(prediction)

test_img1=image.load_img("C:/Users/Sana Kanwal/Downloads/Compressed/23_29dataset/single_prediction/cat_or_dog_1.jpg",target_size=(64,64))
test_img1=image.img_to_array(test_img1)
test_img1=np.expand_dims(test_img1,axis=0)
result=classifier.predict(test_img1)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
    print(prediction)
else:
    prediction='cat'
    print(prediction)

'''history obj is used to  record training metrics for each epoch'''
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
from skimage.io import imread, imshow
'''Predicted Image'''
image = imread('C:/Users/Sana Kanwal/Downloads/Compressed/23_29dataset/single_prediction/cat_or_dog_1.jpg', as_gray=False)
imshow(image)
plt.title('Predicted image {}'.format(prediction))

'''Parameter Tuning'''
filenames = os.listdir("C:/Users/Sana Kanwal/Downloads/Compressed/dataset/training_set")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
print(df.head())

'''Parameter tuning'''
def classifier1():
    classifier=Sequential()
    ''' First Conv and Max pool Hidden Layer'''
    classifier.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    '''Second  Conv and Max pool Hidden Layer'''
    classifier.add(Conv2D(16,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    '''Flatten Layer'''
    classifier.add(Flatten())

    '''Fully Connected Layer'''
    classifier.add(Dense(units=128,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))

    '''Compile'''
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier1
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold

kf=KFold(n_splits=2)
builtclassifier1=KerasClassifier(build_fn=classifier1)
#parameters={'epochs':[5,10],'step_per_epoch':[500,1000]}
parameters={'epochs':[10,20]}
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
gr=GridSearchCV(estimator=builtclassifier1,param_grid=parameters,cv=kf,scoring='accuracy',refit=False)
X=df['filename']
y=df['category']
gr.fit(X,y)
print(gr.best_params_)
print(gr.best_score_)