import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import SqueezeNet
import DataLoad

from numpy import moveaxis

(X,Y) = DataLoad.load_data()  # loads in the images and puts them in a tupple where X is the image and Y the label, only training images
X = X / 255.0  # normalizing the pixel values.

# Reshape data dimensions from (3, 48, 48) to (48, 48, 3)
X = moveaxis(X, 1, 3)

# Create the model. Give the dimensions of the input data as parameter
model = SqueezeNet.SqueezeNet(43, inputs=(48, 48, 3))  # 43 is number of classes, 48 by 48 is the image size

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
                X,
                Y,
                epochs=2,  # this was 4, made it smaller
                steps_per_epoch=200,  # this was 400
                validation_split=0.33)  # we can remove this after we added the test data
                # validation_data=(test_images, test_labels))  # we can add this after we added the test data


# Evaluate the model
# plt.plot(history.history['accuracy'], label='accuracy')
# #plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print(test_acc)
