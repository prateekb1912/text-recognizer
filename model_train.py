from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


from text_recognizer import TextRecognizerModel
from utils import letters
from generator import DataGenerator

import pandas as pd

img_h = 32
img_w = 170
img_c = 1

num_classes = len(letters)+1
batch_size = 64
max_length = 15


LR = 1e-4
EPOCHS = 8

modelObj = TextRecognizerModel(img_h, img_w)
model_input, prediction, model = modelObj.train()

print(model.summary())

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

train_paths = train_df['filename']
train_labels = train_df['label']

val_paths = val_df['filename']
val_labels = val_df['label']

early_stop = EarlyStopping(
    monitor='val_loss', patience=2, restore_best_weights=True)

model_chk_pt = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                               save_best_only=False, save_weights_only=True, verbose=0, mode='auto', period=2)

optimizer = Adam(learning_rate=LR, decay=LR/EPOCHS)

train_gen = DataGenerator(
    train_paths,
    img_w,
    img_h,
    batch_size,
    len(train_paths),
    train_labels
)

val_gen = DataGenerator(
    val_paths,
    img_w,
    img_h,
    batch_size,
    len(val_paths),
    val_labels
)

train_gen.build_data()

test_func = K.function([model_input], [prediction])

train_num_batches = int(train_gen.n/batch_size)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
              optimizer=optimizer)


model.fit_generator(
    generator = train_gen.next_batch(),
    steps_per_epoch = train_num_batches,
    epochs = 5,
    callbacks = [early_stop, model_chk_pt],
    validation_data = val_gen.next_batch(),
    validation_steps = int(val_gen.n/batch_size)
)

model.save('model_weights.h5')