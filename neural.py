import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Start importing the dataset
dataset = keras.datasets.fashion_mnist

# load the dataset
((train_imgs, train_labels),(test_imgs, test_labels)) = dataset.load_data()

# exploring data
print(f'Test image shape: {train_imgs.shape}')
print(f'Train image shape: {test_imgs.shape}')

classification_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
'Bag', 'Ankle boot']

# ploting images from the dataset
plt.figure(figsize=(10,10))
for img in range(10):
    plt.subplot(2, 5, img+1)
    plt.imshow(train_imgs[img],cmap='Greys')
    plt.title(classification_labels[train_labels[img]])
plt.show()

# color bar of the image
plt.imshow(train_imgs[0], cmap='Greys')
plt.colorbar()
plt.show()

# reducing complexity - image normalization
train_imgs = train_imgs/255.0

# creating the model
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(256, activation=tensorflow.nn.relu),
        keras.layers.Dense(128, activation=tensorflow.nn.relu),
        # keras.layers.Dense(64, activation=tensorflow.nn.relu),
        keras.layers.Dense(10, activation=tensorflow.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_imgs, train_labels, epochs=5, validation_split=0.2)

# testing the model
test =  model.predict(test_imgs)
print(test[0])
print('Prediction Result: ',np.argmax(test[0])) #argmax returns the indices of the maximum values along an axis.
print('Expected Result: ', test_labels[0])

test_loss, test_accuracy = model.evaluate(test_imgs, test_labels)
print('Test Loss', test_loss)
print('Test Accuracy', test_accuracy)
