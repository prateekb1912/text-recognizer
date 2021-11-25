from keras import backend as K


modelObj = TextRecognizerModel(img_h, img_w, img_c)

model_input, prediction, model = modelObj.train()

print(model.summary())