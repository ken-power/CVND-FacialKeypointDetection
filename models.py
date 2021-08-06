import torch.nn as nn


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
        self.activation_layer1 = nn.ELU()
        self.pooling_layer1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.normalization_layer1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.activation_layer2 = nn.ELU()
        self.pooling_layer2 = nn.MaxPool2d(kernel_size=4, stride=3)
        self.normalization_layer2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.activation_layer3 = nn.ELU()
        self.pooling_layer3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.normalization_layer3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.activation_layer4 = nn.ELU()
        self.pooling_layer4 = nn.MaxPool2d(kernel_size=6, stride=3)
        self.normalization_layer4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p=0.4)

        self.flatten1 = nn.Flatten()
        self.dense1 = nn.Linear(1024 * 1 * 1, 512)
        self.activation_layer5 = nn.ELU()
        self.dropout5 = nn.Dropout(p=0.1)

        self.dense2 = nn.Linear(512, 256)
        self.activation_layer6 = nn.ELU()
        self.dropout6 = nn.Dropout(p=0.1)

        self.dense3 = nn.Linear(256, 136)

    def forward(self, x):
        """
        Define the feedforward behavior of this model.

        :param x: the input image

        :return: a modified x, having gone through all the layers of the model
        """
        x = self.conv_layer1(x)
        x = self.activation_layer1(x)
        x = self.pooling_layer1(x)
        x = self.normalization_layer1(x)
        x = self.dropout1(x)

        x = self.conv_layer2(x)
        x = self.activation_layer2(x)
        x = self.pooling_layer2(x)
        x = self.normalization_layer2(x)
        x = self.dropout2(x)

        x = self.conv_layer3(x)
        x = self.activation_layer3(x)
        x = self.pooling_layer3(x)
        x = self.normalization_layer3(x)
        x = self.dropout3(x)

        x = self.conv_layer4(x)
        x = self.activation_layer4(x)
        x = self.pooling_layer4(x)
        x = self.normalization_layer4(x)
        x = self.dropout4(x)

        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.activation_layer5(x)
        x = self.dropout5(x)

        x = self.dense2(x)
        x = self.activation_layer6(x)
        x = self.dropout6(x)

        x = self.dense3(x)

        # return a modified x, having gone through all the layers of the model
        return x
