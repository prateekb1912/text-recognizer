from pandas.core.arrays.sparse import dtype
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

import numpy as np

data = []
labels = []

img_h = 32
img_w = 170

with open("annotations.txt", "r") as f:
    for img_file in f:
        file = dict()
        file['filename'] = img_file.strip()

        img = load_img(file['filename'], target_size = (img_h, img_w))
        img = img_to_array(img)

        folder, label, ext = img_file.split('_')
        file['label'] = label.lower()
        print(label)

        labels.append(label)
        data.append(img)

data =  np.array(data, dtype='float32')
labels = np.array(labels)

# We have sampled 27798 images from the MJ Synthetic Dataset
# Now, we will create train, validation and test sets from the whole available dataset
# Train set = 17798 images, Validation set = 5000 images, Test set = 5000 images

trainX = data[:17798]
valX = data[17998:22998]
testX = data[22998:]

trainY = labels[:17798]
valY =labels[17998:22998]
testY = labels[22998:]

np.save('train_data.npy', trainX)
np.save('train_labels.npy', trainY)

np.save('val_data.npy', valX)
np.save('val_labels.npy', valY)

np.save('test_data.npy', testX)
np.save('test_labels.npy', testY)