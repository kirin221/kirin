# In[1]:


import os
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers 
from keras import backend as K
from keras import callbacks
import time
import matplotlib.pyplot as plt

start = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:

train_data_dir = 'data/train'
validation_data_dir = 'data/train'


# In[3]:

nb_train_samples = 96 #276 jumlah total gambar yang ditrain
nb_validation_samples = 96 #jumlah total gambar yang ditrain
batch_size = 1 #jumlah perfoto yang ditraining


# In[4]:

img_width, img_height = 221, 480 #ukuran gambar yang di trainning

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

K.clear_session()
# In[5]:

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)  


# In[6]:

test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[7]:

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[8]:

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# In[9]:

model = Sequential()
#model.add(Conv2D(12, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(25, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Flatten())
#model.add(Dense(180, activation='relu')) 
#model.add(Dropout(0.5))
#model.add(Dense(128, activation='relu')) 
#model.add(Dropout(0.5))
#model.add(Dense(34, activation='softmax')) 

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation = "softmax")) #jumlah output


# In[10]:

model.compile(loss='categorical_crossentropy',
              #optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
              optimizer='adadelta',
              metrics=['accuracy'])


# In[11]:

epochs = 30


# In[12]:

history=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# In[13]:

# Evaluate model
score = model.evaluate_generator(validation_generator, nb_validation_samples//batch_size, workers=12, use_multiprocessing=False)
print ("Evaluate generator results:")
print ("Evaluate loss: " + str(score[0]))
print ("Evaluate accuracy: " + str(score[1]))


# In[14]:

model.save('ETA.h5')

end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minute")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper right')
plt.show()



