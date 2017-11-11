import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

NUM_CLASSES = 43
TRAINED_MODEL_PRESENT = False

class TrafficSignClassfier:
    def __init__(self, image_dir_path):
        self.image_dir_path = image_dir_path
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model =  None
    # function for reading the images
    # arguments: path to the traffic sign data, for example './GTSRB/Training'
    # returns: list of images, list of corresponding labels 
    def readTrafficSigns(self):
        '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

        Arguments: path to the traffic sign data, for example './GTSRB/Training'
        Returns:   list of images, list of corresponding labels'''
        images = [] # images
        labels = [] # corresponding labels
        # loop over all 42 classes
        for c in range(0,43):
            prefix = self.image_dir_path + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
                labels.append(row[7]) # the 8th column is the label
            gtFile.close()
        return images, labels


    def build_model(self):

        try:
            self.model = self.load_model()
            TRAINED_MODEL_PRESENT = True
            return
        except OSError as err:
            print("OS error: {0}".format(err))

        
        model = Sequential()
        print(self.x_train[0].shape)
        model.add(Conv2D(32, (3,3), input_shape=(self.x_train[0].shape)))
        BatchNormalization(axis = -1)
        model.add(Activation('relu'))
        model.add(Conv2D(64,(3,3)))
        BatchNormalization(axis = -1)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3, 3)))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())

        #Fully connected neural network
        model.add(Dense(512))
        BatchNormalization()
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(NUM_CLASSES))
        model.add(Activation('softmax'))

        model.summary()

        self.model = model

    def pre_process_images(self, images, to_color = 'GRAY', size=(60,60)):

        processed_imgs = []
        for image in images:
            image = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            processed_imgs.append(image)

        return processed_imgs

    def train_test_split(self, images, labels, test_size):

        labels = np_utils.to_categorical(labels, NUM_CLASSES)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(images), labels, test_size=test_size)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1)



    def fit_model(self):
        
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        gen = ImageDataGenerator()

        train_gen = gen.flow(self.x_train, self.y_train, batch_size = 64)
        test_gen = gen.flow(self.x_test, self.y_test, batch_size = 64)
        
        self.model.fit_generator(train_gen, steps_per_epoch=6000//64, epochs=4, 
                    validation_data=test_gen, validation_steps=4000//64)

        self.save_model()

    def predict(self):
    	print(self.model.predict(self.x_test))

    def save_model(self):
        self.model.save('traffic_sign_classifier.h5')

    def load_model(self):
        return load_model('traffic_sign_classifier.h5')


if __name__ == "__main__":
    
    traffic_sign_classifier = TrafficSignClassfier('/home/ashok/Desktop/Learning_Exercises/german traffic sign classifier/GTSRB/Final_Training/Images')
    images, labels = traffic_sign_classifier.readTrafficSigns()
    pre_procesed_images = traffic_sign_classifier.pre_process_images(images)
    traffic_sign_classifier.train_test_split(pre_procesed_images, labels, 0.3)
    traffic_sign_classifier.build_model()
    
    if TRAINED_MODEL_PRESENT:
    	traffic_sign_classifier.predict()
    else:
    	traffic_sign_classifier.fit_model()