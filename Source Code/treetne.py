import numpy as np
from joblib import load
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D
from keras_preprocessing.image import ImageDataGenerator, os
from keras.models import Sequential, Model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

clf = load("BeetleModel.joblib")

datagen = ImageDataGenerator(rescale=1. / 255.)

generator = datagen.flow_from_directory(
    "./test/",
    target_size=(416, 416),
    batch_size=1,
    class_mode="sparse",
    shuffle=False)

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

# generate and save features, labels and respective filenames
steps = 1
X = model.predict_generator(generator, 1)
tf

l = []
for i in enumerate(X):
    l.append(i)

"""X = [generator[0]]
"""
X_A = []
for i in l:
    X_A = np.concatenate([X_A[0], i], 1)

"""X_A = np.concatenate([X[0], X[1], X[2], X[3], X[4]], 1)"""

"""
X = [X[0], X[1], X[2], X[3], X[4], X_A]
"""
y_pred = clf.predict(X_A)
acc = accuracy_score(1, y_pred) * 100
print(y_pred)
print(acc)