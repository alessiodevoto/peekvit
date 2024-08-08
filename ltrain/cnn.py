import torch
import torch.nn as nn
import math

def calculate_output_size(input_size, filter_size, stride, padding):
    return math.floor((input_size - filter_size + 2 * padding) / stride) + 1

def calculate_flattened_size(height, width):
    # # Define your layer configurations
    layers = [
        {'type': 'conv', 'filter_size': 3, 'stride': 1, 'padding': 0, 'num_filters': 8},
        {'type': 'pool', 'filter_size': 2, 'stride': 2},
        {'type': 'conv', 'filter_size': 3, 'stride': 1, 'padding': 0, 'num_filters': 16},
        {'type': 'pool', 'filter_size': 2, 'stride': 2},
        # {'type': 'conv', 'filter_size': 3, 'stride': 1, 'padding': 0, 'num_filters': 16},
        # {'type': 'pool', 'filter_size': 4, 'stride': 4},
    ]
    for layer in layers:
        if layer['type'] == 'conv':
            height = calculate_output_size(height, layer['filter_size'], layer['stride'], layer['padding'])
            width = calculate_output_size(width, layer['filter_size'], layer['stride'], layer['padding'])
        elif layer['type'] == 'pool':
            height = calculate_output_size(height, layer['filter_size'], layer['stride'], 0)
            width = calculate_output_size(width, layer['filter_size'], layer['stride'], 0)
    
    # Assume the depth (number of filters) after the last conv layer
    num_filters = layers[-2].get('num_filters', 1)
    
    # Flatten size
    flattened_size = height * width * num_filters
    return flattened_size

# Define the PyTorch model
class CustomCNN(nn.Module):
    def __init__(self, flattened_size, output_size):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        # self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, output_size)  # Assuming binary classification
        self.flattened_size = flattened_size  

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        #x = self.pool3(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the PyTorch model
class CustomTransformerTopology(nn.Module):
    def __init__(self, trnasformer_layer, height, width, output_size):
        super(CustomTransformerTopology, self).__init__()
        
        input_dim = trnasformer_layer.norm1.weight.shape[0]
        self.map_input = nn.Linear(height, input_dim)
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_size)  # Assuming binary classification
        self.trnasformer_layer = trnasformer_layer

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x = self.map_input(x)

        x = self.trnasformer_layer(x)
        x = x.sum(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# # Define your layer configurations
# layers = [
#     {'type': 'conv', 'filter_size': 3, 'stride': 1, 'padding': 0, 'num_filters': 32},
#     {'type': 'pool', 'filter_size': 2, 'stride': 2},
#     {'type': 'conv', 'filter_size': 3, 'stride': 1, 'padding': 0, 'num_filters': 64},
#     {'type': 'pool', 'filter_size': 2, 'stride': 2}
# ]

# # Initial dimensions
# height = 196
# width = 132

# # Calculate the flattened size
# flattened_size = calculate_flattened_size(height, width, layers)
# print(f'Flattened size: {flattened_size}')