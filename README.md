<img src="img/logo.png" alt="logo" width="250" heigth="200"/>

The purpose of this repository is my self practice of Deep Learning techniques. Exercises and Project used in the online course provided from Alura and other resources indicated on this file.

I am a student that is learning, let me know if you find any errors,the code is inspired from examples and exercises found in the course.

# What have I learned from this Project?

1. The project is all about start working with machine learning tools for image classification. We will use the famous Fashion MNIST data-set and try to create a model with Keras and Tensor-flow that can predict what is the correct label for the image.

The dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels).

2. Initially we need to appreciate the difference between Machine Learning and Deep Learning. The difference relies on how the data is structured and presented to the algorithm, ML most of time requires a well structured and predefined labeled features to work while DL relies on the multiple layers of the Neural networks to define such features.

Using ML to try to predict the label of the images from the Fashion MNIST would be too complicated to do manually(defining the features), the best way to identify what features can help us label the data-set is use Deep Learning.

"In artificial neural networks, hidden layers are required if and only if the data must be separated non-linearly."


3. This is an extremely simple project whereas most of the code is about using the Tensor-flow/Keras API functions, the trade-off here is that there is a lots of abstraction but it simplifies a lot of the work. Later on I will be working on a more "from scratch" approach.

4. Data Preprocess

We need to simplify the complexity of the data, to do so we will reduce the range of the pixel values to a 0-1 interval
by dividing the Numpy Array by 255.0

5. Tensors
Tensors are data structures used by Neural Networks, they are mathematical objects that generalize scalars, vectors and matrices to higher dimensions.
Tensors are multiple dimensions arrays.

6. The Model

Neural Networks modeling requires configuring the different layers of the network and then compiling the model.

* keras.Flatten
The first layer will reformat the data, transforming the 2d array (28x28) to a one dimension array of 784 pixels.

* keras.Dense
Densely connected or fully connected neural layers that will ultimately return an array of 10 probability score that will indicate the probability that the current image belong to one of the 10 different classes/labels

* Compiling the Model
- Loss: Cross Entropy Loss
- Metrics
- Optimizer: adam

* Training the model


7. Activation Functions

Also known as Transfer Function, help us to determine the output of the network node, maps the resulting values in between 0 to 1.

* Linear Activation Functions
Not useful for complexity or data with various parameters
* Non-linear
Easier for the model to generalize with variety of data.

For this project the activation function used what the Rectified Linear Unit, any value that is negative becomes 0 and any positive value is equal to a positive value.

Softmax is used to define the probabilities of the output for each category.

## Resources that I used to learn about this fun topic:
- Online Course: Deep Learning part 1, Alura.
- https://deeplizard.com/learn/video/gZmobeGL0Yg
- https://hackernoon.com/deep-learning-vs-machine-learning-a-simple-explanation-47405b3eef08
- https://www.tensorflow.org/tutorials/keras/classification
- https://towardsdatascience.com/quick-ml-concepts-tensors-eb1330d7760f
- https://towardsdatascience.com/quick-ml-concepts-tensors-eb1330d7760f
- http://scipy-lectures.org/advanced/image_processing/
- https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
- https://en.wikipedia.org/wiki/Softmax_function
- https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
- https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
- https://medium.com/@dibyadas/visualizing-different-normalization-techniques-84ea5cc8c378
- https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e
- https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
- https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5 
