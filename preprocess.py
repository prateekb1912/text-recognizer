import pandas as pd
from PIL import Image

dataset = []

with open("annotations.txt", "r") as f:
    for img_file in f:
        file = dict()
        file['filename'] = img_file.strip()

        # Convert the images to grayscale
        img = Image.open(file['filename']).convert('L')
        img.save(file['filename'])

        folder, label, ext = img_file.split('_')
        file['label'] = label.lower()
        print(label)

        dataset.append(file)

df = pd.DataFrame(dataset)

# We have sampled 27998 images from the MJ Synthetic Dataset
# Now, we will create train, validation and test sets from the whole available dataset
# Train set = 17798 images, Validation set = 5000 images, Test set = 5000 images

train_df = df[:17798]
val_df = df[17998:22998]
test_df = df[22998:]

train_df.to_csv("train.csv")
val_df.to_csv("val.csv")
test_df.to_csv("test.csv")

print(train_df.head())
print(test_df.head())