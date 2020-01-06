import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# Start importing the dataset
dataset = keras.datasets.fashion_mnist

# load the dataset, return 4 Numpy arrays
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
test_imgs = test_imgs/255.0

# creating the model
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(256, activation=tensorflow.nn.relu),
        keras.layers.Dense(128, activation=tensorflow.nn.relu),
        keras.layers.Dropout(0.2),
        # keras.layers.Dense(64, activation=tensorflow.nn.relu),
        keras.layers.Dense(10, activation=tensorflow.nn.softmax)])

# trying different weights
# bias_dense_layer = model.layers[1].get_weights()[1]
# random_weights_dense_layer = np.random.rand(784, 256)
# model.layers[1].set_weights([random_weights_dense_layer, bias_dense_layer])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_imgs, train_labels, epochs=5, validation_split=0.2)


# saving trained model
# model.save('model_epochs5_nodes4.h5')
saved_model = load_model('model_epochs5_nodes4.h5')

# ploting accuracy per epochs
print(history.history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy per epochs')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['training', 'validation'])
plt.show()

# ploting loss per epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss per epochs')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['training', 'validation'])
plt.show()


# testing the model
test =  model.predict(test_imgs)
print(test[0])
print('Prediction Result: ',np.argmax(test[0])) #argmax returns the indices of the maximum values along an axis.
print('Expected Result: ', test_labels[0])

# testing saved model
test_saved_model = saved_model.predict(test_imgs)
print('Prediction Result: ',np.argmax(test_saved_model[0])) #argmax returns the indices of the maximum values along an axis.
print('Expected Result: ', test_labels[0])

test_loss, test_accuracy = model.evaluate(test_imgs, test_labels)
print('Test Loss', test_loss)
print('Test Accuracy', test_accuracy)

# Model summary, weights and bias
model_summary = model.summary()
print('Printing the weights and bias: ', model.layers[1].get_weights())
