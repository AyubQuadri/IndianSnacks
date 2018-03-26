import cv2               # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from matplotlib import pyplot as plt
TRAIN_DIR ='train_images'
TEST_DIR= 'test_images'
IMG_SIZE=280
def label_img(img):
    word = img.split("(")[-2]
    if word=='samosa ': return 0
    elif word=='kachori ': return 1
    elif word=='aloo_paratha ': return 2
    elif word=='idli ': return 3
    elif word=='jalebi ': return 4
    elif word=='tea ': return 5
    elif word=='paneer_tikka ': return 6
    elif word=='dosa ': return 7
    elif word=='omlet ': return 8
    elif word=='poha ': return 9
    
def load_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data_k.npy', training_data)
    return training_data

def load_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),np.array(label)])
    shuffle(testing_data)
    np.save('test_data_k.npy', testing_data)
    return testing_data

def load_train_data_NPY():
    data = np.load('train_data.npy')
    return data

def load_test_data_NPY():
    data = np.load('test_data.npy')
    return data


def predict_imgs(model,No_of_Images,Imgs_np):
    fig=plt.figure()
    for num,data in enumerate(Imgs_np[:No_of_Images]):

        
        img_num = data[1]
        img_data = data[0]
        y = fig.add_subplot(3,No_of_Images/3,num+1)
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,1)
        #model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
        
        if np.argmax(model_out) == 0: str_label='samosa'
        elif np.argmax(model_out) == 1: str_label='kachori'
        elif np.argmax(model_out) == 2: str_label='aloo_paratha'
        elif np.argmax(model_out) == 3: str_label='idli'
        elif np.argmax(model_out) == 4: str_label='jalebi'
        elif np.argmax(model_out) == 5: str_label='tea'
        elif np.argmax(model_out) == 6: str_label='paneer_tikka'
        elif np.argmax(model_out) == 7: str_label='dosa'
        elif np.argmax(model_out) == 8: str_label='omlet'
        elif np.argmax(model_out) == 9: str_label='poha'
            
        y.imshow(orig,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()