import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# define image height and width, channels, batch size
img_height = 416
img_width = 416
channels = 3
batch_size = 32

# define image generator without augmentation
train = "./SnipDat/Train/"
val = "./SnipDat/Val/"

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=train, target_size=(img_height, img_width))

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=val, target_size=(img_height, img_width))

datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    train,
    color_mode = "rgb",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse')

x_train, y_train = next(train_generator)
x_test, y_test = next(train_generator)

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

model2.summary()

batch_size = 32
datagen = ImageDataGenerator(rescale=1. / 255.)

generator2 = datagen.flow_from_directory(
    train,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False)

y_pred = model2.predict_generator(generator2, 705//batch_size, workers=4)

for i in y_pred[:10]:
    print(float(i))

model2.predict_classes(x_test)
print(np.count_nonzero(y_pred == y_test)/len(y_test))


plt.plot(plt.hist.history["acc"])
plt.plot(plt.hist.history['val_acc'])
plt.plot(plt.hist.history['loss'])
plt.plot(plt.hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()

from keras.preprocessing import image
img = image.load_img("./Test/test1.jpg", target_size=(416, 416))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)

from keras.models import load_model
saved_model = load_model("vgg16_1.h5")
output = saved_model.predict(img)

for i, f in enumerate(generator2.filenames):
    print(f, y_pred[i])