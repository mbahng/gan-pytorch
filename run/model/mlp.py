import torch.nn as nn
from torch import Tensor

class MLP(nn.Module): 

  def __init__(self, idim = 10, hdim = 20, odim = 5): 
    super().__init__()
    self.fc1 = nn.Linear(idim, hdim) 
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hdim, odim)
    self._init_weights()

  def forward(self, x: Tensor): 
    x = self.fc1(x) 
    x = self.relu1(x) 
    x = self.fc2(x) 
    return x

  def _init_weights(self): 
    ...
