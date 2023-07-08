import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from preprocess_image import preprocess

path='dataset'

IMG_SIZE = 96

def create_train_data():
    training_data = []
    label=0
    for (dirpath,dirnames,filenames) in os.walk(path):
        for dirname in dirnames:
            print(dirname)
            for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
                for file in files:
                        actual_path=path+"/"+dirname+"/"+file
                        # label=label_img(dirname)
                        path1 =path+"/"+dirname+'/'+file
                        img=cv2.imread(path1)
                        img = preprocess(img)
                        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
                        training_data.append([np.array(img),label])
            label=label+1
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    print(training_data)
    return training_data

create_train_data()