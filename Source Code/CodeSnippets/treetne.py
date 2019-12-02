import h5py
import numpy as np
from joblib import load
from keras import Model
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D
from keras_preprocessing.image import ImageDataGenerator


# Load the SVM we trained with joblib
from sklearn.metrics import confusion_matrix, classification_report

clf = load("../BeetleModel.joblib")

# build the VGG16 network and extract features after every MaxPool layer
# We just have to set up the model so it's the same as the feature extraction model
model = VGG16(weights='imagenet', include_top=False)

c1 = model.layers[-16].output
c1 = GlobalAveragePooling2D()(c1)

c2 = model.layers[-13].output
c2 = GlobalAveragePooling2D()(c2)

c3 = model.layers[-9].output
c3 = GlobalAveragePooling2D()(c3)

c4 = model.layers[-5].output
c4 = GlobalAveragePooling2D()(c4)

c5 = model.layers[-1].output
c5 = GlobalAveragePooling2D()(c5)


model = Model(inputs=model.input, outputs=(c1, c2, c3, c4, c5))

# define image generator
datagen = ImageDataGenerator(rescale=1. / 255.)
generator = datagen.flow_from_directory(
    './test/',
    target_size=(416, 416),
    batch_size=3,
    class_mode="sparse",
    shuffle=False)

# generate and save features and labels
steps = 4
batch_size = 3
X = model.predict_generator(generator, steps)
Y = np.concatenate([generator.next()[1] for i in range(0, generator.samples, batch_size)])

# Get the shape for CSV
x = []
for n, i in enumerate(X):
    x.append(i)

cvscores, sqrnscores = [], []

X_a = np.concatenate([x[0], x[1], x[2], x[3], x[4]], 1)


# Predict vs the CSV
y_pred = clf.predict(X_a)

print("Class", y_pred)

Z = np.load("/home/cmazur/Documents/SpiderTree/Source Code/datasets/D1/features/Y.npy")
labels = sorted(list(set(Z)))
strLab = []
for i in labels:
    strLab.append(str(i))

print("Confusion matrix: ")
print("Labels: {0}".format(",".join(strLab)))
print(confusion_matrix(Y, y_pred, labels = strLab))

print("Classification report: ")
print(classification_report(Y, y_pred))


#
# # Load an image from file with load_img, resize to standard size
# image = load_img('./test/Asilidae/test1.jpg', target_size=(416, 416))
#
# # convert the image pixels to a numpy array with img_to_array
# imagearr = img_to_array(image)
#
# # reshape data for the model - take out RGB
# imagers = imagearr.reshape((1, imagearr.shape[0], imagearr.shape[1], imagearr.shape[2]))
#
# # prepare the image for the VGG model with preprocess_input
# imagepp = preprocess_input(imagers)
#
# # predict the probability across all output classes
# yhat = clf.predict(imagepp)
#
# # convert the probabilities to class labels with decode_predictions
# label = decode_predictions(yhat)
# # retrieve the most likely result, e.g. highest probability
# labelout = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (labelout[1], labelout[2]*100))
