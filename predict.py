from utils import encode_words_labels, words_from_labels
from text_recognizer import TextRecognizerModel
import itertools

import cv2
import numpy as np
import pandas as pd

model = TextRecognizerModel()
pred_model = model.predict()

pred_model.load_weights('model_weights.h5')

def decode_label(out):
    """
    Takes the predicted ouput matrix from the Model and returns the output text for the image
    """
    # out : (1, 42, 37)
    # discarding first 2 outputs of RNN as they tend to be garbage 
    out_best = list(np.argmin(out[0,2:], axis=1))

    print(len(out_best))

    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

    print(out_best)

    outstr = words_from_labels(out_best)

    return outstr

def test_data_single_image_Prediction(model,test_img_path):
    """
    Takes the best model, test data image paths, test data groud truth labels and pre-processes the input image to 
    appropriate format for the model prediction, takes the predicted output matrix and uses best path decoding to 
    generate predicted text and prints the Predicted Text Label, Time Taken for Computation
    """
    
    test_img=cv2.imread(test_img_path)
    test_img_resized=cv2.resize(test_img,(128, 128))
    test_image=test_img_resized[:,:,1]
    test_image=np.expand_dims(test_image,axis=-1)
    test_image=np.expand_dims(test_image, axis=0)
    test_image=test_image/255
    
    print(test_image.shape)

    model_output = model.predict(test_image)
    print(model_output)
    
    predicted_output=decode_label(model_output)
    print("Predicted Text in the Image: ", predicted_output) 

test_img = pd.read_csv('test.csv').iloc[0]
test_data_single_image_Prediction(pred_model, test_img['filename'])