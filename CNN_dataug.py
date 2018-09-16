import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.cross_validation import train_test_split

K.set_image_dim_ordering('tf')

np.random.seed(237)

# Loading the data
train_orig = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# Train and Validation split
labels = train_orig['label']
train = train_orig.drop('label', axis=1)
X_train, X_val, y_train, y_val = train_test_split(train, labels, test_size=0.10, random_state=42)

# Normalizing the datasets
X_train = X_train.astype('float32')/255.
X_train = X_train.values.reshape(X_train.shape[0],28,28,1).astype('float32')

X_val = X_val.astype('float32')/255.
X_val = X_val.values.reshape(X_val.shape[0],28,28,1).astype('float32')

X_test = X_test.astype('float32')/255.
X_test = X_test.values.reshape(X_test.shape[0],28,28,1).astype('float32')

# One hot encoding of labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# K_classes
K_classes = 10

# Let's make a model
convnet3 = Sequential()

# Conv2D*2 -> MaxPool -> Dropout #1
convnet3.add(Conv2D(32, (3,3), input_shape=(28,28,1), padding='same', activation='relu'))
#convnet3.add(Conv2D(32, (3,3), padding='same', activation='relu'))
convnet3.add(MaxPooling2D(pool_size=(2, 2)))
convnet3.add(Dropout(0.10))

# Conv2D*2 -> MaxPool -> Dropout #2
#convnet3.add(Conv2D(64, (3,3), strides=(2,2), padding='same', activation='relu'))
convnet3.add(Conv2D(64, (3,3), padding='same', activation='relu'))
convnet3.add(MaxPooling2D(pool_size=(2, 2)))
convnet3.add(Dropout(0.10))

# Conv2D*2 -> MaxPool -> Dropout #3
#convnet3.add(Conv2D(, (3,3), strides=(2,2), padding='same', activation='relu'))
convnet3.add(Conv2D(128, (3,3), padding='same', activation='relu'))
convnet3.add(MaxPooling2D(pool_size=(2, 2)))
convnet3.add(Dropout(0.10))

# Flatten -> Dense -> Dense -> Out
convnet3.add(Flatten())
convnet3.add(Dense(256, activation='relu'))
convnet3.add(Dense(128, activation='relu'))
convnet3.add(Dense(K_classes, activation='softmax'))

# Stochastic gradient descent
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
convnet3.compile(loss='categorical_crossentropy', 
                 optimizer=sgd, 
                 metrics=['accuracy'])

# Data Augmentation
augdata = ImageDataGenerator(featurewise_center=False,
          samplewise_center=False,
          featurewise_std_normalization=False,
          samplewise_std_normalization=False,
          zca_whitening=False,
          rotation_range=False,
          zoom_range = 0.2, 
          width_shift_range=0.15,
          height_shift_range=0.15,
          horizontal_flip=False,
          vertical_flip=False)

augdata.fit(X_train)

# Fit convnet on this data
convnet3_fit = convnet3.fit_generator(augdata.flow(X_train, y_train, batch_size=50), epochs=30, validation_data=(X_val, y_val), verbose=2)

# make predictions on test
convnet3_test_preds = convnet3.predict(X_test)

# predict as the class with highest probability
convnet3_test_preds = np.argmax(convnet3_test_preds, axis = 1)

# put predictions in pandas Series
convnet3_test_preds = pd.Series(convnet3_test_preds, name='label')

# Add 'ImageId' column
convnet3_for_csv = pd.concat([pd.Series(range(1,28001), name='ImageId'), 
                              convnet3_test_preds], axis=1)

# write to csv for submission
convnet3_for_csv.to_csv('convnet4_3layers163264128_Dropout_keras_30epochs.csv', index=False)
