import tensorflow as tf

from numpy import moveaxis

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import SqueezeNet

# import Cifar dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Reshape dataset
print(train_images.shape)
train_images = moveaxis(train_images, 3, 1)
test_images = moveaxis(test_images, 3, 1)
print(train_images.shape)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


model = SqueezeNet.SqueezeNet(10, inputs=(3, 32, 32))

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data=(test_images, test_labels))

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
