[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

I completed this project as part of Udacity's [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) program.

The goal of this project is to combine computer vision techniques and deep learning architectures to build a facial keypoint detection system. 

Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include facial tracking, facial pose recognition, facial filters, and emotion recognition. The facial keypoint detector is able to look at any image, detect faces, and predict the locations of facial keypoints on each face. 

Examples of keypoints detected on faces are shown here:

![Facial Keypoint Detection][image1]

# Project Specification

## Define a Convolutional Neural Network 

CRITERIA | MEETS SPECIFICATIONS
:--- | :---
Define a CNN in `models.py` | Define a convolutional neural network with at least one convolutional layer, i.e. `self.conv1 = nn.Conv2d(1, 32, 5)`. The network should take in a grayscale, square image.

The CNN is defined in the file [models.py](models.py).

## Define the Network Architecture 

CRITERIA | MEETS SPECIFICATIONS
:--- | :---
Define the `data_transform` for training and test data | Define a `data_transform` and apply it whenever you instantiate a DataLoader. The composed transform should include: rescaling/cropping, normalization, and turning input images into torch Tensors. The transform should turn any input image into a normalized, square, grayscale image and then a Tensor for your model to take it as input.
Define the loss and optimization functions | Select a loss function and optimizer for training the model. The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem.
Train the CNN | Train your CNN after defining its loss and optimization functions. You are encouraged, but not required, to visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. Save your best trained model.
Answer questions about model architecture  | After training, all 3 questions about model architecture, choice of loss function, and choice of batch_size and epoch parameters are answered.
Visualize one or more learned feature maps | Your CNN "learns" (updates the weights in its convolutional layers) to recognize features and this criteria requires that you extract at least one convolutional filter from your trained model, apply it to an image, and see what effect this filter has on an image.
Answer question about feature visualization | After visualizing a feature map, answer: what do you think it detects? This answer should be informed by how a filtered image (from the criteria above) looks.

The CNN architecture is defined in the notebook [Define the Network Architecture](2.%20Define%20the%20Network%20Architecture.ipynb).

## Facial Keypoint Detection

CRITERIA | MEETS SPECIFICATIONS
:--- | :---
Detect faces in a given image | Use a Haar cascade face detector to detect faces in a given image.
Transform each detected face into an input Tensor | You should transform any face into a normalized, square, grayscale image and then a Tensor for your model to take in as input (similar to what the `data_transform` did in Notebook 2).
Predict and display the keypoints on each face | After face detection with a Haar cascade and face pre-processing, apply your trained model to each detected face, and display the predicted keypoints for each face in the image.

Facial Keypoint Detection is implemented in the notebook [Facial Keypoint Detection, Complete Pipeline](3.%20Facial%20Keypoint%20Detection,%20Complete%20Pipeline.ipynb).

# References

* Udacity, 2021. [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) program lectures, notes, exercises.
* PyTorch Documentation. [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam).
* Kingma, D.P. and Ba, J., 2014. [Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980).
* Jason Brownlee, 2017. [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/).
* Casper Hansen, 2019. [Optimizers Explained - Adam, Momentum and Stochastic Gradient Descent](https://mlfromscratch.com/optimizers-explained/#/).
* PyTorch Documentation. [`torch.nn.MSELoss`](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html?highlight=mse#torch.nn.MSELoss)
* Jason Brownlee, 2019. [Loss and Loss Functions for Training Deep Learning Neural Networks](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/).
* Jason Brownlee, 2020. [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/).
* Agarwal, N., Krohn-Grimberghe, A. and Vyas, R., 2017. [Facial key points detection using deep convolutional neural network-NaimishNet](https://arxiv.org/abs/1710.00977). arXiv preprint arXiv:1710.00977.
* PyTorch documentation. [`torch.nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html).
* Clevert, D.A., Unterthiner, T. and Hochreiter, S., 2015. _Fast and accurate deep network learning by exponential linear units (elus)_. [arXiv preprint arXiv:1511.07289](https://arxiv.org/pdf/1511.07289.pdf).
* Pedamonti, D., 2018. Comparison of non-linear activation functions for deep neural networks on MNIST classification task. [arXiv preprint arXiv:1804.02763](https://arxiv.org/pdf/1804.02763.pdf).
* Jason Brownlee, 2019a. [A Gentle Introduction to Dropout for Regularizing Deep Neural Networks](). Machine Learning Mastery.
* Jason Brownlee, 2019b. [A Gentle Introduction to Pooling Layers for Convolutional Neural Networks](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/). Machine Learning Mastery.
* Jason Brownlee, 2020. [Dropout Regularization in Deep Learning Models With Keras](https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/). Machine Learning Mastery.
* Amar Budhiraja, 2016. [Dropout in (Deep) Machine learning](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5). Medium.
* Alessio Gozzoli, 2018. [Practical Guide to Hyperparameters Optimization for Deep Learning Models](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/). FloydHub.
* Francois Chollet, 2018. Deep Learning with Python, _Chapter 5: Deep Learning for Computer Vision_. Manning Publications Co. 

