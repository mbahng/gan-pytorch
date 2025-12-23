import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self, in_channels: int):
    super().__init__()
    # Convolutional layers
    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)  # 28x28x32
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14x64
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 7x7x64
    
    # Fully connected layers
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)
    
    # Dropout for regularization
    self.dropout = nn.Dropout(0.5)
      
  def forward(self, x):
    # Conv layer 1
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)  # 14x14x32
    
    # Conv layer 2
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)  # 7x7x64
    
    # Conv layer 3
    x = F.relu(self.conv3(x))
    
    # Flatten
    x = x.view(-1, 64 * 7 * 7)
    
    # Fully connected layers
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x

