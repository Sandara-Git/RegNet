import typing
import torch
import torch.nn as nn
from torch.functional import Tensor

class ConvRNN(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super().__init__()
        self.kernel_size = kernel_size
         # Create a 2D convolutional layer for processing input x
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True  
        )         
        
        # Store the number of intermediate channels                                                                      
        self.intermediate_channels = intermediate_channels

    def forward(self, x:Tensor, state:Tensor) -> Tensor:
         # Concatenate the input x with the state along the channel dimension
        x = torch.cat([x, state], dim=1)
          # Pass the concatenated input through the convolutional layer
        x = self.conv_x(x)
        # Apply the tanh activation function to the output
        return torch.tanh(x)
  # Method to initialize the state
    def init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> Tensor:
         # Extract the spatial dimensions
        width, height = spatial_dim
         # Initialize the state tensor with zeros
        y = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
         # Initialize the state tensor with zeros
        return y


class ConvLSTMCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super(ConvLSTMCell, self).__init__()
        """conv_x  has a valid padding by:
        setting padding = kernel_size // 2
        hidden channels for h & c = intermediate_channels
        """
        self.intermediate_channels = intermediate_channels
        # Create a 2D convolutional layer for processing input x
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels *  4,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> typing.Tuple:
        """
        c and h channels = intermediate_channels so  a * c is valid
        if the last dim in c not equal to a then a has been halved
        """
        # Extract the cell and hidden states from the input state
        c, h = state
        # Move the hidden state h and cell state c to the same device as input x
        h = h.to(device=x.device)
        c = c.to(device=x.device)
        # Concatenate the input x with the hidden state h
        x = torch.cat([x, h], dim=1)
        # Pass the concatenated input through the convolutional layer
        x = self.conv_x(x)
        # Split the output of the convolutional layer into four parts
        a, b, g, d = torch.split(x, self.intermediate_channels, dim=1)
          # Apply activation functions to a, b, and g
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        g = torch.sigmoid(g)
        # Apply the tanh function to d
        d = torch.tanh(d)
        # Update the cell state c and hidden state h
        c =  a * c +  g * d
        h = b * torch.tanh(c)
        # Return the updated cell and hidden states
        return c, h
 # Method to initialize cell and hidden states
    def init_states(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> typing.Tuple:
         # Extract spatial dimensions
        width, height = spatial_dim
         # Initialize cell state c and hidden state h with zeros
        c = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        h = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return c, h

class ConvGRUCell(nn.Module):
    def __init__(self, x_in:int, intermediate_channels:int, kernel_size:int=3):
        super(ConvGRUCell, self).__init__()
        """
        x dim = 3
        b * h = b & h should have same dim
        hidden channels for h = intermediate_channels
        """
        # Create a 2D convolutional layer for processing input x
        self.conv_x = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels * 2,
            kernel_size=kernel_size, padding= kernel_size // 2, bias=True
        )
        # Create another 2D convolutional layer for processing y
        self.conv_y = nn.Conv2d(
            x_in + intermediate_channels, intermediate_channels,
            kernel_size=kernel_size, padding=kernel_size // 2,  bias=True
        )
        self.intermediate_channels = intermediate_channels
    def forward(self, x:Tensor, state:typing.Tuple[Tensor, Tensor]) -> Tensor:
         # Clone the input tensor x
        y = x.clone()
        # Extract the hidden state from the input state
        _, h = state
        h = h.to(device=x.device)
        # Concatenate the input x with the hidden state h
        x = torch.cat([x, h], dim=1)
         # Pass the concatenated input through the convolutional layer
        x = self.conv_x(x)
        # Split the output of the convolutional layer into two parts
        a, b = torch.split(x, self.intermediate_channels, dim=1)
        a = torch.sigmoid(a)
        b = torch.sigmoid(b)
        # Concatenate the cloned input y with b * h
        y = torch.cat([y, b * h], dim=1)
         # Pass the concatenated tensor through the second convolutional layer and apply the tanh function
        y = torch.tanh(self.conv_y(y))
         # Update the hidden state h
        h = a * h + (1 - a) * y
        # Return None (as there's no output other than the updated hidden state) and the updated hidden state
        return None, h

    def _init_state(self, batch_size:int, channels:int, spatial_dim:typing.Tuple) -> Tensor:
        width, height = spatial_dim
        h = torch.zeros(batch_size, channels, width, height, device=self.conv_x.weight.device)
        return h


if __name__ == "__main__":
      # Create a random tensor
    x = torch.randn((4, 10 , 3, 24, 24))
    # Select a slice of the tensor for input x
    x = x[:, 1, :, :, : ]
    # Initialize a tensor h with zeros
    h= torch.zeros((4, 128, 24, 24))
    # Initialize a tensor h with zeros
    s = ConvLSTMCell(3, 128, 3)
     # Apply the ConvLSTMCell to input x and initial state (h, h)
    y = s(x, (h, h))
    # Print the shape of the first element of y
    print(y[0].shape)



