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

classification_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
'Bag', 'Ankle boot']

# ploting images from the dataset
plt.figure(figsize=(10,10))
for img in range(10):
    plt.subplot(2, 5, img+1)
    plt.imshow(train_img[img],cmap='gray')
    plt.title(classification_labels[train_label[img]])
plt.show()
