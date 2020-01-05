import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt

# Start importing the dataset
dataset = keras.datasets.fashion_mnist

# load the dataset
((train_img, train_label),(test_img, test_label)) = dataset.load_data()
# print(train_img)


print(f'Test image shape: {train_img.shape}')
print(f'Train image shape: {test_img.shape}')

# ploting image at position 0 from training dataset
plt.imshow(train_img[0])
plt.show()
