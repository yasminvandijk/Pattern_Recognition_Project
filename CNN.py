from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load in the images and put them in a tuple where X is the image and Y the label
import DataLoad
(X, Y) = DataLoad.load_data()

# Reshape data dimensions from (3, 48, 48) to (48, 48, 3)
from numpy import moveaxis
X = moveaxis(X, 1, 3)

# Normalize pixel values to be between 0 and 1
X = X / 255.0

# Split data into training and validation (test) set
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(43, activation='softmax'))
model.summary()

# Compile and train the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Test the model on the dataset
history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))


# Evaluate the model: accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('CNN + DCGAN accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model: loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('CNN + DCGAN loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
