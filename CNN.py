import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import SqueezeNet
import DataLoad

(X,Y) = DataLoad.load_data() #loads in the images and puts them in a tupple where X is the image and Y the label, only training images
X = X / 255.0 #normalizing the pixel values.

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0

#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#               'dog', 'frog', 'horse', 'ship', 'truck']

# Create the model. Give the dimensions of the input data as parameter
model = SqueezeNet.SqueezeNet(43, inputs=(48, 48, 3)) #43 is number of classes, 48 by 48 is the image size, original was 32,32,3 and 10 classes

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
                X,
                Y,
                epochs=2, #this was 4, made it smaller
                steps_per_epoch=200) #this was 400
                #validation_data=(test_images, test_labels)) #validation set it not yet loaded in


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
