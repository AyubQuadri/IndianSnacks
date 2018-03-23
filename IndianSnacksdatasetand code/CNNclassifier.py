import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm

from timeit import default_timer as timer
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt
import indiansnacksdataset as data

#train_data = data.load_train_data()
test_data = data.load_test_data()

#train_data = np.load('train_data_K.npy')
#test_data = np.load('test_data_k.npy')
batch_size = 128
num_classes = 10
epochs = 10
IMG_SIZE =280

# input image dimensions

# load train-data 

start = timer()

train = test_data[:-200]
val = test_data[-200:]

X_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y_train = [i[1] for i in train]


val_x = np.array([i[0] for i in val]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
val_y = [i[1] for i in val]




#plt.imshow(X_train[0].shape)
X_train[0].shape

input_shape = (IMG_SIZE, IMG_SIZE, 1)

# convert the data to the right type
X_train = X_train.astype('float32')
val_x = val_x.astype('float32')
X_train /= 255
val_x /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(val_x.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
Y_train = np_utils.to_categorical(Y_train, num_classes)
val_y = np_utils.to_categorical(val_y, num_classes)


modelCNN = Sequential()
modelCNN.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
modelCNN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

modelCNN.add(Conv2D(64, (5, 5), activation='relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Conv2D(64, (5, 5), activation='relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Conv2D(128, (5, 5), activation='relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Conv2D(64, (5, 5), activation='relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))

modelCNN.add(Flatten())

modelCNN.add(Dense(1024, activation='relu'))
modelCNN.add(Dense(num_classes, activation='softmax'))

modelCNN.summary()

modelCNN.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])



history = modelCNN.fit(X_train, Y_train, 
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data= (val_x,val_y))

score = modelCNN.evaluate(val_x, val_y, verbose=0)


end = timer()
print(end - start)

print(history.history['acc'])
print(history.history['val_acc'])

print(history.history['loss'])
print(history.history['val_loss'])


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# save the model in Saved_Model Folder
 
# serialize model to JSON
model_json = modelCNN.to_json()
with open("Saved_Model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Saved_Model/model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('Saved_Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Saved_Model/CNNModelIndian.h5")
print("Loaded model from disk")

#predict using the test model
data.predict_test(loaded_model, 12)

