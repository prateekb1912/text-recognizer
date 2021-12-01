from os import name
from numpy.core.fromnumeric import shape
from pandas.core.arrays.sparse import dtype
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D
from keras.layers import Activation, Bidirectional
from keras.layers import BatchNormalization, Dropout
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers.recurrent import LSTM
from tensorflow.keras.applications import MobileNetV2

from utils import ctc_loss_function, letters

num_classes = len(letters)+1
batch_size = 64
max_length = 25


class TextRecognizerModel():
    """
        Builds the text recognizer model layer by layer
    """

    def __init__(self, img_h = 32, img_w = 170):
        """
            Initializes the model with the image size passed as arguments
        """
        self.input_shape = (img_w, img_h, 1)
        self.labels = None
        self.label_length = 0
        self.input_length = 0
        self.loss = None
        self.model = None

        self.model_input = Input(
            shape=self.input_shape, name='img_input', dtype='float32')

        self.add_cnn_layers()
        self.reshape_model()
        self.add_rnn_layers()

    def add_cnn_layers(self):
        """
            Defining the architecture of the first half of the model which processes
            the input image and consisting of convolution layers.
        """

        self.model = Conv2D(64, (3, 3), padding='same', name='conv1',
                            kernel_initializer='he_normal')(self.model_input)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)
        self.model = MaxPooling2D(pool_size=(2, 2), name='max1')(self.model)

        self.model = Conv2D(128, (3, 3), padding='same', name='conv2',
                            kernel_initializer='he_normal')(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)
        self.model = MaxPooling2D(pool_size=(2, 2), name='max2')(self.model)

        self.model = Conv2D(256, (3, 3), padding='same', name='conv3',
                            kernel_initializer='he_normal')(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)
        self.model = Conv2D(256, (3, 3), padding='same', name='conv4',
                            kernel_initializer='he_normal')(self.model)
        self.model = Dropout(0.35)(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)
        self.model = MaxPooling2D(pool_size=(1, 2), name='max3')(self.model)

        self.model = Conv2D(512, (3, 3), padding='same', name='conv5',
                            kernel_initializer='he_normal')(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)
        self.model = Conv2D(512, (3, 3), padding='same',
                            name='conv6')(self.model)
        self.model = Dropout(0.35)(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)
        self.model = MaxPooling2D(pool_size=(1, 2), name='max4')(self.model)

        self.model = Conv2D(512, (2, 2), padding='same',
                            kernel_initializer='he_normal', name='con7')(self.model)
        self.model = Dropout(0.25)(self.model)
        self.model = BatchNormalization()(self.model)
        self.model = Activation('relu')(self.model)

    def reshape_model(self):
        """
            Reshape the model to transition from CNN to RNN
        """
        self.model = Reshape(target_shape=((42, 1024)),
                             name='reshape')(self.model)
        self.model = Dense(
            64, activation='relu', kernel_initializer='he_normal', name='dense1')(self.model)

    def add_rnn_layers(self):
        """
            Define the RNN layers to go from image to text for predicition
        """
        self.model = (Bidirectional(LSTM(256, return_sequences=True,
                                         kernel_initializer='he_normal'), merge_mode='sum'))(self.model)
        self.model = (Bidirectional(LSTM(256, return_sequences=True,
                                         kernel_initializer='he_normal'), merge_mode='sum'))(self.model)

        # Transform the RNN output to character activations
        # Now, we will be using the functional API as there will be multiple outputs now

        self.model = Dense(num_classes, kernel_initializer='he_normal',
                           name='dense2')(self.model)

        # Ouput predicted text, take input actual label and their respective lengths for loss
        # computation
        self.prediction = Activation('softmax', name='softmax')(self.model)
        
        self.labels = Input(name='ground_truth_labels', shape=[
            max_length], dtype='float32')
        self.input_length = Input(
            name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(
            name='label_length', shape=[1], dtype='int64')

        # CTC loss function
        self.loss = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')([
            self.prediction, self.labels,
            self.input_length, self.label_length])

    def train(self):
        return self.model_input, self.prediction, Model(inputs=[self.model_input, self.labels,
                                                                self.input_length, self.label_length], outputs=self.loss)
    
    def predict(self):
        return Model(inputs = [self.model_input], outputs = self.prediction)
