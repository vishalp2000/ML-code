import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('orangelid.jpg')#replace with camera feed frames
#imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#plt.imshow(imgrgb)
#plt.show()

resize = tf.image.resize(img, (256,256))
#plt.imshow(resize.numpy().astype(int))
#plt.show()
#keras.models.
new_model = load_model('models/imageclassifier.h5')
new_model.predict(np.expand_dims(resize/255, 0))
new_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
img_width, img_height = 320, 240
img = image.load_img('orangelid.jpg', target_size=(img_width, img_height))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)

#images = np.vstack([x])
#classes = new_model.predict(images, batch_size=10)
yhat = new_model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5: 
    print(f'Predicted class is yellow')
else:
    print(f'Predicted class is Orange')
