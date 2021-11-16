import pandas as pd
from PIL import Image

dataset = []

with open("annotations.txt", "r") as f:
    for img_file in f:
        file = dict()
        file['filename'] = img_file.strip()

        folder, label, ext = img_file.split('_')
        file['label'] = label.lower()

        dataset.append(file)