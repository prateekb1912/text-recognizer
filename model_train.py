from keras import backend as K

from utils import ctc_loss_function, letters

from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, MaxPooling2D
from keras.layers import Activation, Bidirectional
from keras.layers import BatchNormalization, Dropout
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import Reshape
from keras.models import Model
from keras.layers.recurrent import LSTM


img_h = 32
img_w = 170
img_c = 1

num_classes = len(letters)+1
batch_size = 64
max_length = 15


class TextRecognizerModel():
    """
        Builds the text recognizer model layer by layer
    """

    def __init__(self, img_h, img_w, img_c):
        """
            Initializes the model with the image size passed as arguments
        """
        self.input_shape = (img_h, img_w, img_c)
        self.model = Input(shape=self.input_shape,
                           name='img_input')

        self.add_layers()

    def add_cnn_layers(self):
        """
            Defining the architecture of the first half of the model which processes
            the input image and consisting of convolution layers.
        """

        self.model.add(Conv2D(64, (3, 3),
                              padding='same', name='conv2d_1'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool_1'))

        self.model.add(Conv2D(128, (3, 3),
                              padding='same', name='conv2d_2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool_2'))

        self.model.add(Conv2D(256, (3, 3), padding='same', name='conv2d_3'))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(1, 2), name='max3'))

        self.model.add(Conv2D(512, (3, 3),
                              padding='same', name='conv2d_4'))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), name='max_pool_4'))

        self.model.add(Conv2D(512, (3, 3),
                              padding='same', name='conv2d_5'))
        self.model.add(Dropout(0.3))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        
    def reshape_model(self):
        """
            Reshape the model to transition from CNN to RNN
        """
        self.model.add(Reshape(target_shape=((42, 1024)), name='reshape'))
        self.model.add(Dense(64, activation='relu', name='dense1'))


def textRecognizerModel(stage, drop_out_rate=0.35):

    # RNN layer
    model = Bidirectional(LSTM(256, return_sequences=True,
                          kernel_initializer='he_normal'), merge_mode='sum')(model)
    model = Bidirectional(LSTM(256, return_sequences=True,
                          kernel_initializer='he_normal'), merge_mode='concat')(model)

    # transforms RNN output to character activations:
    model = Dense(num_classes, kernel_initializer='he_normal',
                  name='dense2')(model)
    y_pred = Activation('softmax', name='softmax')(model)

    labels = Input(name='ground_truth_labels', shape=[
                   max_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # CTC loss function
    loss_out = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    if stage == 'train':
        return model_input, y_pred, Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
    else:
        return Model(inputs=[model_input], outputs=y_pred)


model_input, y_pred, img_text_recog = textRecognizerModel('train')

test_func = K.function([model_input], [y_pred])
img_text_recog.summary()
