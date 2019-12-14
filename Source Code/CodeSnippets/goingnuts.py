import numpy as np
from keras import optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from PIL import ImageFile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from keras.models import load_model
from sklearn.svm import LinearSVC

ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.utils import  check_X_y

# Helper
def kFoldFit(X, Y, model, cvscores):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=555)
    splits = []
    for train, test in kfold.split(X, Y):
        # Train the model
        model.fit(X[train], Y[train])

        # Save your models as you go it costs nothing
        model.save("model2trainnumber" + str(train) + ".h5")

        # Check the stats
        y_pred = model.predict(Y[test])
        acc = accuracy_score(Y[test], y_pred) * 100
        cvscores.append(acc)
        splits.append((Y[test], y_pred))
####



# define image height and width, channels, batch size
img_height = 416
img_width = 416
channels = 3
batch_size = 8

# define image generator without augmentation
train = "./SnipDat/Train/"
val = "./SnipDat/Val/"

trdata = ImageDataGenerator(rescale=1./255)
traindata = trdata.flow_from_directory(directory=train, color_mode="rgb", target_size=(img_height, img_width),
                                       batch_size=batch_size, class_mode='sparse')

tsdata = ImageDataGenerator(rescale=1./255)
testdata = tsdata.flow_from_directory(directory=val, color_mode="rgb", target_size=(img_height, img_width),
                                      batch_size=batch_size, class_mode='sparse')


# Generate data using a bunch of wigglified images
datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

generator = datagen.flow_from_directory(
    "../datasets/D1/images/",
    color_mode = "rgb",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')

x_train, y_train = next(traindata)
x_test, y_test = next(testdata)


# Load our model to save a whole bunch of work lmao
Model_Loaded_at_Start = False
try:
    model2 = load_model("model2thewholething.h5")
    print("Model loaded")
    Model_Loaded_at_Start = True

except IOError:
    print("Gotta load the whole thing again ugh")

    model2 = VGG16(weights='imagenet', include_top=False)

    c1 = model2.layers[-16].output
    c1 = GlobalAveragePooling2D()(c1)

    c2 = model2.layers[-13].output
    c2 = GlobalAveragePooling2D()(c2)

    c3 = model2.layers[-9].output
    c3 = GlobalAveragePooling2D()(c3)

    c4 = model2.layers[-5].output
    c4 = GlobalAveragePooling2D()(c4)

    c5 = model2.layers[-1].output
    c5 = GlobalAveragePooling2D()(c5)

    model2 = Model(inputs=model2.input, outputs=(c1, c2, c3, c4, c5))

model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model2.summary()

batch_size = 16
# Get X, Y, Text Labels. Note that X can be big so we gonna save it.
try:
    X = load("goingnutsX.joblib")

    X_Train = load("goingnutsX_Train.joblib")

    X_Test = load("goingnutsX_Test.joblib")
    print("Got X's")
except IOError:
    print("Couldn't get X's")
    X = model2.predict_generator(generator, 882 / batch_size, workers = 4)
    dump(X, "goingnutsX.joblib")

    X_Train = model2.predict_generator(traindata, 705 / batch_size, workers=4)
    dump(X, "goingnutsX_Train.joblib")

    X_Test = model2.predict_generator(testdata, 177 / batch_size, workers=4)
    dump(X, "goingnutsX_Test.joblib")

try:
    Y = load("goingnutsY.joblib")
    Y_Train = load("goingnutsY_Train.joblib")
    Y_Test = load("goingnutsY_Test.joblib")
    print("Got Y's")

except IOError:
    print("Couldn't get Y's")
    Y = generator.classes
    Y_Train = traindata.classes
    Y_Test = testdata.classes
    # Y needs to be an array.
    np.save("goingnutsY.npy", Y)
    np.save("goingnutsY_Train.npy", Y_Train)
    np.save("goingnutsY_Test.npy", Y_Test)


try:
    Labels = load("goingnutsLabels.joblib")
    Labels_Train = load("goingnutsLabels_Train.joblib")
    Labels_Test = load("goingnutsLabels_Test.joblib")
    print("Got Labels")

except IOError:
    print("Couldn't get Labels")
    Labels = generator.class_indices
    Labels_Train = traindata.class_indices
    Labels_Test = testdata.class_indices
    # Labels needs to be an array.
    Labels = np.array(list(Labels))
    Labels_Train = np.array(list(Labels_Train))
    Labels_Test = np.array(list(Labels_Test))
    np.save("goingnutsLabels.npy", Labels)
    np.save("goingnutsLabels_Train.npy", Labels_Train)
    np.save("goingnutsLabels_Test.npy", Labels_Test)


# Fit model to training data
Model_Got_Trained = False
try:
    model2 = load_model("model2thewholething_afterTraining.h5")
    cvscores = load("cvscores.joblib")
    Model_Got_Trained = True
    print("Model from after training was loaded")

except IOError:
    print("Gotta fit this model again")
    cvscores = []
    # model2.fit(X_Train[0], Y_Train)
    clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
    x_a = np.reshape(([X_Train[0], X_Train[1], X_Train[2], X_Train[3], X_Train[4]]), -1)
    x_train = x_a.reshape(705, -1)
    x_test = x_a.reshape(705, -1)
    clf.fit(x_train, Y_Train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(Y_Test, y_pred) * 100
    cvscores.append(acc)

    # for n, x in enumerate(X):
    #     print()
    #     if n == 5:
    #         print("fused features across all conv blocks")
    #     else:
    #         print("conv block", n + 1)
    #
    #     print("without normalization")
    #     kFoldFit(x, Y, model2, cvscores)
    #
    #     print("with square root normalization")
    #     x = np.sqrt(np.abs(x)) * np.sign(x)
    #     kFoldFit(x, Y, model2, cvscores)

    # Dump the scores for checkin' later if you want em
    dump(cvscores, "cvscores.joblib")
    Model_Got_Trained = True

print("Accuracy score averaged across 10 kfolds %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# don't forget to save the thing damn lol
model2.save("model2thewholething_afterTraining.h5")
model2.save("model2thewholething.h5")
print("Model saved")

# Check predictions against classes - working on it
#y_classes = y_pred.argmax(axis=-1)

# Test the sucker against completely new data!
from keras.preprocessing import image
img = image.load_img("./Test/test1.jpg", target_size=(416, 416))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

output = model2.predict(img)
print(output)


# plt.plot(plt.hist.history["acc"])
# plt.plot(plt.hist.history['val_acc'])
# plt.plot(plt.hist.history['loss'])
# plt.plot(plt.hist.history['val_loss'])
# plt.title("model accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
# plt.show()