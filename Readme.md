Training a neural network using PyTorch on MNIST Dataset
In this project, We are using PyTorch to train a deep learning multi-class classifier on this dataset and test how the trained model performs on the test samples.

#### Neural Network Architecture:
In this project, 
* we will create a convolutional neural network that contains:
  *   convolutional,
  *   linear,
  *   max-pooling, and
  *   dropout layers.
  *   Log-Softmax is used for the final layer and
  *   ReLU is used as the activation function for all the other layers. And
  *   The model is trained using an Adadelta optimizer with a fixed learning rate of 0.5.

#### MNIST Dataset
For this exercise, we will be using the famous MNIST dataset [5], 
  * which is a sequence of images of handwritten postcode digits, zero through nine, with corresponding labels.
  * The MNIST dataset consists of 60,000 training samples and 10,000 test samples,
  * where each sample is a grayscale image with 28 x 28 pixels. PyTorch also provides the MNIST dataset under its Dataset module.

  * MNIST, or the Modified National Institute of Standards and Technology database, is a collection of handwritten digits that's commonly used for training and testing image processing systems and machine learning: 
    * What it is
      A large database of 70,000 grayscale images of handwritten digits, each 28x28 pixels in size 
    * How it's used
      A standard benchmark for evaluating image classification algorithms. It's also used as a "hello world" example by data scientists. 
    * How it was created
      A derivative work from the original NIST Special Database 1 and Special Database 3, which contain images of handwritten digits written by high school students and US Census Bureau employees, respectively 
    * Who created it
      Yann LeCun of Courant Institute, NYU and Corinna Cortes of Google Labs, New York hold the copyright 
    * Where to find it
      You can find the MNIST dataset on: 
      Yann LeCun's website: Includes the train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, and t10k-labels-idx1-ubyte.gz files 
      Hugging Face: Includes the ylecun/mnist dataset 
      GitHub: Includes the cvdfoundation/mnist repository 

