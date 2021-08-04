import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I  # Used to initialize the weights of your Net


class Net(nn.Module):
    """
    Define the convolutional neural network architecture
    """

    def __init__(self):
        """
        Define all the layers of this CNN, the only requirements are:
            1. This network takes in a square (same width and height), grayscale image as input
            2. It ends with a linear layer that represents the keypoints

        It is suggested to make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        """
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1)


        self.fc1 = nn.Linear(43264, 1000)
        self.fc2 = nn.Linear(1000, 136)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.4)


    def forward(self, x):
        """
        Define the feedforward behavior of this model.

        :param x: x is the input image and, as an example, here you may choose to include a pool/conv step:
                  x = self.pool(F.relu(self.conv1(x)))

        :return: a modified x, having gone through all the layers of the model
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(self.pool(F.relu(self.conv4(x))))

        # flatten
        x = x.view(x.size(0), -1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # a modified x, having gone through all the layers of the model, should be returned
        return x
