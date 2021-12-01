import shutil
import os

RootDir1 = "90kDICT32px"
TargetFolder = "Images"
num_images= 0

for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
    if num_images > 60000:
        break
    for name in files:
        if name.endswith('.jpg'):
            num_images += 1
            print(f"Found {num_images}")
            SourceFolder = os.path.join(root,name)
            shutil.copy2(SourceFolder, TargetFolder) #copies csv to new folder