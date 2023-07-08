import cv2        
import numpy as np
import os   
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

train_data = np.load('train_data.npy', allow_pickle=True)
X = train_data[:,0]
y = to_categorical(train_data[:,1])
X = np.array([i for i in X])

X = X[:,:,:, np.newaxis]
y = y[:,:,np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333)
print(X_train.shape)