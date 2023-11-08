import torch
import torch.nn as nn
import torchmetrics as tm
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

import argparse

import pytorch_lightning as pl
from torch.functional import Tensor
from typing import Tuple, Dict, List
from conv_rnns import ConvGRUCell, ConvLSTMCell


from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from cifar10_datamodule import Cifar10DataModule
from components_datamodule import ComponentsDataModule

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

momentum = 0.9
max_epochs = 30
batch_size = 32

class SELayer(nn.Module):
    def __init__(self, in_dim:int, reduction_factor:int=8) -> None:
        super(SELayer, self).__init__()
        # Create an adaptive average pooling layer that outputs a 1x1 tensor
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # Create a sequential module that consists of two linear layers and activation functions
        self.sequential= nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction_factor, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction_factor, in_dim, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x:Tensor):
        # Extract batch size, number of channels, height, and width from the input tensor
        B, C, _, _ = x.shape
         # Apply global average pooling and reshape the result to be of shape (B, C)
        y = self.global_avg_pool(x).view(B, C)
         # Pass the reshaped tensor through the sequential module and reshape the result back to the original shape
        y = self.sequential(y).view(B, C, 1, 1)
        # Scale the input tensor x by the computed values in y
        x = x * y.expand_as(x)
        return x



class rnn_regulated_block(nn.Module):
    def __init__(self, h_dim, in_channels, intermediate_channels, rnn_cell, identity_block=None, stride=1):
        super(rnn_regulated_block, self).__init__()
        #print(f'In channels {in_channels} | Intermediate channels: {intermediate_channels} ')
        self.stride = stride
        self.h_dim = h_dim
        self.identity_block = identity_block
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU()

        self.rnn_cell = rnn_cell
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, 
                               kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        #Multiply intermediate_channels by 2, torch.cat([hidden_state, x])
        self.conv3 = nn.Conv2d(h_dim + intermediate_channels, intermediate_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(intermediate_channels)

        self.conv4 = nn.Conv2d(intermediate_channels, intermediate_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm2d(intermediate_channels * 4)

        # Create a Squeeze-and-Excitation layer
        self.se_layer = SELayer(intermediate_channels * 4, reduction_factor=8)
        # Define the dimensions for downsampling the state
        downsample_dim = h_dim if isinstance(rnn_cell, ConvGRUCell) else h_dim * 2
        #Cell state dim remains constant but aspect ratio of the feature map is variable
        self.downsample_state = nn.LazyConv2d(downsample_dim, kernel_size=3, stride=stride, padding=1)


    def forward(self, x:torch.Tensor, state:Tuple) -> Tuple:
        y = x.clone()
        # Perform the necessary operations for the first convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
    # Apply the rnn_cell to the processed tensor x and the input state    
        c, h = self.rnn_cell(x, state)
        
        #print(f'Block running {x.shape}')
# Perform the necessary operations for the second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Perform the necessary operations for the first convolutional layer
        x = torch.cat([x, h], dim=1)
# Perform the necessary operations for the third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
# Perform the necessary operations for the fourth convolutional layer
        x = self.conv4(x)
        x = self.bn4(x)
 # Apply the Squeeze-and-Excitation layer to the tensor x
        x = self.se_layer(x)
# If an identity block is provided, apply the identity block and downsample the state accordingly
        if self.identity_block is not None:
            y = self.identity_block(y)
            if c is not None:
                s = torch.cat([c, h], dim=1)
                s = self.downsample_state(s)
                c, h = torch.split(s, self.h_dim, dim=1)
            else:
                h = self.downsample_state(h)
 # Add the cloned tensor y to tensor x and apply the ReLU activation function
        x += y
 # Return the cell state, hidden state, and the ReLU-activated tensor x 
        return c, h, self.relu(x)


class RegNet(pl.LightningModule):
    def __init__(self, regulated_block:nn.Module, in_dim:int, h_dim:int, intermediate_channels:int,
                 classes:int=3, cell_type:str='gru', layers:List=[3, 3, 3], config=None):
        super(RegNet, self).__init__()
        self.layers = layers
        self.classes = classes
        self.intermediate_channels = intermediate_channels
        self.h_dim = h_dim
        self.cell_type = cell_type
        #self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(in_dim, self.intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.intermediate_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((3, 3) , padding=1, stride=2)
        self.cell = ConvGRUCell if cell_type == 'gru' else ConvLSTMCell

        self.rnn_cells = nn.ModuleList()
        self.regulated_blocks = nn.ModuleList()
        num_layers = len(layers)
        
        #64, 256, 512, 1025
        
        c_in = self.intermediate_channels
        # Create and append rnn_cells based on the number of layers
        for layer in range(num_layers):
            self.rnn_cells.append(self.cell(c_in, h_dim, kernel_size=3))
            c_in = c_in * 4 if layer == 0 else c_in * 2
           # print(f"RNN Cel cl {layer}:", self.rnn_cells[-1])
        # Create and append regulated_blocks based on the number of layers
        for layer in range(num_layers):
            # Set stride and channels based on the layer
            stride = 2
            channels = self.intermediate_channels // 2
            
            if layer < 1:
                stride = 1
                channels = self.intermediate_channels

            # Create an identity block
            identity_block = nn.Sequential(
                nn.Conv2d(self.intermediate_channels, channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )
            # Create and append the regulated block
            self.regulated_blocks.append(
                regulated_block(
                    self.h_dim, self.intermediate_channels, channels,
                    self.cell(channels, h_dim , kernel_size=3),
                    identity_block, stride
                )
            )

            self.intermediate_channels = channels * 4
           # print(f"Regulated Block {layer}:", self.regulated_blocks[-1])

            for block in range(layers[layer] - 1):
                self.regulated_blocks.append(
                    regulated_block(
                        self.h_dim, self.intermediate_channels, channels,
                        self.cell(channels, h_dim, kernel_size=3)
                    )
                )   
                # print(f"Regulated Block {layer}-{block}:", self.regulated_blocks[-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(classes)
# Initialize metrics for accuracy
        self.val_accuracy = tm.Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy = tm.Accuracy(task='multiclass', num_classes=10)
        self.train_accuracy = tm.Accuracy(task='multiclass', num_classes=10)
# Store the configuration and hyperparameters
        self.config = config
        
        self.save_hyperparameters()


    def forward(self, x) -> Tensor:
        # Apply the necessary operations for the initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.max_pool(x)
        B, _, H, W = x.shape
        # Initialize the tensors for the hidden state and cell state
        h = torch.zeros(B, self.h_dim, H, W)
        c = torch.zeros(B, self.h_dim, H, W) if self.cell_type != 'gru' else None
        # Apply the first rnn_cell to the input tensor x and the initial states
        c, h = self.rnn_cells[0](x, (c, h))
        
        layer_idx = 0
        block_sum = 0
        
        print(len(self.regulated_blocks))
        print('e')
        x_fmaps=[]
      # Iterate through the regulated_blocks and apply the blocks accordingly
        for i, block in enumerate(self.regulated_blocks):
            c, h, x = block(x, (c, h))
            block_sum += 1
            if layer_idx < len(self.layers) - 1 and block_sum == self.layers[layer_idx]:
                #print(f'Block {i}, {x.shape}, {h.shape}, {block_sum}')
                c, h = self.rnn_cells[layer_idx + 1](x, (c, h))
                x_fmaps.append(x)
                layer_idx += 1
                block_sum = 0
# Apply average pooling, flatten the tensor, and pass it through the output layer
        # x = self.avg_pool(x)
        # x = self.flatten(x)
        print(len(x_fmaps))
        print('a')
        # return self.output(x)
        # return self.output(x)
        # print (self.output(x))
        return x_fmaps[0],x_fmaps[1],x_fmaps[2]

if __name__  == "__main__":  
    model=RegNet(rnn_regulated_block,in_dim=3,h_dim=64,intermediate_channels=32,classes=6,cell_type='gru',layers=[1, 1, 3, 3]) 

    # img_path = 'car.jpg'
    image=cv2.imread("car.jpg")
    image=cv2.resize(image,(512,512))

    transform=transforms.ToTensor()
    x=transform(image).unsqueeze(0)
    print(x.shape)
    print('b')
#     img = Image.open(img_path)
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     input_img = preprocess(img).unsqueeze(0)  # Add a batch dimension

# # Pass the preprocessed image through the model
#     output = model(input_img)

    x3,x4,x5=model(x)
    # print(x3.type())torch.FloatTensor
    print(x3.shape)
    print('c')
    # x3_array=x3.detach().numpy()
    # combined_image_x3 = x3_array[0].sum(axis=0)
    # plt.imsave('combined_image_x3.png', combined_image_x3, cmap='viridis', format='png')
    
    print(x4.shape)
    # x4_array=x4.detach().numpy()
    # combined_image_x4 = x4_array[0].sum(axis=0)
    # plt.imsave('combined_image_x4.png', combined_image_x4, cmap='viridis', format='png')
    
    print(x5.shape)