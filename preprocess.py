import pandas as pd
import cv2

dataset = []

num_files = 0

with open("annotations.txt", "r") as f:
    for img_file in f:
        file = dict()
        file['filename'] = img_file.strip()

        num_files += 1
        dir = ''

        if num_files < 40000: dir = 'train'
        elif num_files < 50000: dir = 'val'
        else: dir = 'test'

        # Convert the images to grayscale
        img = cv2.imread(file["filename"])
        img = cv2.resize(img,(128,128))
        img = img[:, :, 1]

        folder, label, ext = img_file.split('_')

        file['label'] = label.upper()

        file['filename'] = f"dataset/{dir}/{label}.jpg"

        print(f"{num_files}. {file['filename']}: {label}")
        cv2.imwrite(file['filename'], img)

        dataset.append(file)

df = pd.DataFrame(dataset)

# We have sampled 60000 images from the MJ Synthetic Dataset
# Now, we will create train, validation and test sets from the whole available dataset
# Train set = 40000 images, Validation set = 10000 images, Test set = 10000 images

train_df = df[:40000]
val_df = df[40000:50000]
test_df = df[50000:]

train_df.to_csv("train.csv", index = False)
val_df.to_csv("val.csv", index = False)
test_df.to_csv("test.csv", index = False)

print(train_df.head())
print(test_df.head())