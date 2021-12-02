import os

DIR = './90kDICT32px'

nums = 0

for root, subdirectories, files in os.walk(DIR):
    for file in files:
        print(os.path.join(root, file))
        with open("annotations.txt", "a") as f:
            f.write(os.path.join(root, file)+"\n")
            print(nums)
            nums += 1
    
    if nums >= 60000:
        break