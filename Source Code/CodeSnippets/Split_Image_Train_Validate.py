import os
import shutil

source = "./SnipDat/Total/"

testDest = "./SnipDat/Val/"
trainDest = "./SnipDat/Train/"

if not os.path.exists(testDest):
    os.makedirs(testDest)
if not os.path.exists(trainDest):
    os.makedirs(trainDest)

for folder in os.listdir(source):
    print(folder)

    if not os.path.exists(testDest + folder):
        os.mkdir(testDest + folder)
    if not os.path.exists(trainDest + folder):
        os.mkdir(trainDest + folder)

    num = len(os.listdir(source + folder))
    t = round(num/5)
    print(num, t)

    for img in os.listdir(source + folder):
        if t > 0:
            print("Test Image: " + img)
            shutil.copy(source + folder + "/" + img, testDest + folder + "/" + img)
            t = t - 1
        else:
            print("Train Image: " + img)
            shutil.copy(source + folder + "/" + img, trainDest + folder + "/" + img)
