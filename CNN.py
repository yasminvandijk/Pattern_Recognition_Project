##CNN
from tensorflow.keras import layers, models
import matplotlib.pyplot as pyplot

# Load in the images and put them in a tuple where X is the image and Y the label
import DataLoad
(X, Y) = DataLoad.load_data()

# Reshape data dimensions from (3, 48, 48) to (48, 48, 3)
from numpy import moveaxis
X = moveaxis(X, 1, 3)

# Normalize pixel values to be between 0 and 1
#X = X / 255.0

# Split data into training and validation (test) set
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42)


# Create the model
def cnn_model():
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

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

model = cnn_model()


##fit the model to the no augmented data
batch_size = 32
epochs = 50
m_no_aug = cnn_model()
m_no_aug.summary()

history_no_aug = m_no_aug.fit(
    train_images, train_labels,
    epochs=epochs, batch_size=batch_size,
    validation_data=(test_images, test_labels))

loss_no_aug, acc_no_aug = m_no_aug.evaluate(test_images,  test_labels)

##results
fig = pyplot.figure()
fig.patch.set_facecolor('white')

pyplot.plot(history_no_aug.history['accuracy'],
         label='train accuracy',
         c='dodgerblue', ls='-')
pyplot.plot(history_no_aug.history['val_accuracy'],
         label='test accuracy',
         c='dodgerblue', ls='--')

pyplot.xlabel('Epoch')
pyplot.ylabel('Accuracy')
pyplot.legend(loc='lower right')
pyplot.show()

##loss results
fig = pyplot.figure()
fig.patch.set_facecolor('white')

pyplot.plot(history_no_aug.history['loss'],
         label='train loss',
         c='dodgerblue', ls='-')
pyplot.plot(history_no_aug.history['val_loss'],
         label='test loss',
         c='dodgerblue', ls='--')

pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.legend(loc='upper right')
pyplot.show()