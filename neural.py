import tensorflow
from tensorflow import keras

# Start importing the dataset
dataset = keras.datasets.fashion_mnist

# load the dataset
((train_img, train_label),(test_img, test_label)) = dataset.load_data()
print(train_img)
