from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras import activations

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (5,5), activation = activations.relu, padding = 'same', input_shape = (96, 96, 1)))
    model.add(Conv2D(32, (5,5), activation = activations.relu, padding = 'same', input_shape = (96, 96, 1)))
    model.add(MaxPooling2D((2,2), strides = 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), activation = activations.relu, padding = 'same', input_shape = (96, 96, 1)))
    model.add(Conv2D(64, (3,3), activation = activations.relu, padding = 'same', input_shape = (96, 96, 1)))
    model.add(MaxPooling2D((2,2), strides = (2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation = activations.relu))
    model.add(Dropout(0.25))
    model.add(Dense(24, activation = activations.softmax))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')
    model.load_weights("D:\project\sign_language_recognize\weights\weights.h5")
    return model