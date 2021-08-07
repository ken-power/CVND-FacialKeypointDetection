import torch.nn as nn
import torch.nn.functional as F


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
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)

        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

        self.fully_connected1 = nn.Linear(in_features=36864, out_features=1000)
        self.fully_connected2 = nn.Linear(in_features=1000, out_features=1000)
        self.fully_connected3 = nn.Linear(in_features=1000, out_features=136)  # 136 = 2 for each of 68 keypoints (x, y) pairs

    def forward(self, x):
        """
        Define the feedforward behavior of this model.

        :param x: the input image

        :return: a modified x, having gone through all the layers of the model
        """
        x = self.dropout1(self.pooling_layer(F.relu(self.conv_layer1(x))))
        x = self.dropout2(self.pooling_layer(F.relu(self.conv_layer2(x))))
        x = self.dropout3(self.pooling_layer(F.relu(self.conv_layer3(x))))
        x = self.dropout4(self.pooling_layer(F.relu(self.conv_layer4(x))))

        # Flattening layer
        x = x.view(x.size(0), -1)

        x = self.dropout5(F.relu(self.fully_connected1(x)))
        x = self.dropout6(F.relu(self.fully_connected2(x)))

        x = self.fully_connected3(x)

        # return a modified x, having gone through all the layers of the model
        return x
