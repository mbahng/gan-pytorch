import torch
import torch.nn as nn

class GAN(nn.Module): 
  """Wrapper class around both discriminator and generator"""

  def __init__(self, data_dim, latent_dim): 
    super().__init__()
    self.generator = self.Generator(data_dim, latent_dim)
    self.discriminator = self.Discriminator(data_dim, latent_dim)

  class Generator(nn.Module): 
    """
    Generator that takes in a prior and maps it to data space
    """
    
    def __init__(self, data_dim=784, latent_dim=100): 
      super().__init__()
      self.data_dim = data_dim 
      self.latent_dim = latent_dim
      self.fc1 = nn.Linear(latent_dim, latent_dim) 
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(latent_dim, data_dim)
      self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
      """
      x should be the sampled tensor from latent dim
      """
      x = self.relu(self.fc1(x)) 
      x = self.relu(self.fc1(x)) 
      x = self.relu(self.fc1(x)) 
      x = self.fc2(x)
      # x = self.sigmoid(self.fc2(x)) 
      return x

    def sample(self, sample_size): 
      return torch.randn(sample_size, self.latent_dim)

    def toggle_grad(self, enable): 
      for param in self.parameters(): 
        param.requires_grad = enable

  class Discriminator(nn.Module): 
    """
    Discriminator MLP that outputs probability that the sample came 
    from the true data generating distribution. 
    """
    
    def __init__(self, data_dim, latent_dim): 
      super().__init__()
      self.data_dim = data_dim
      self.fc1 = nn.Linear(data_dim, latent_dim) 
      self.fc3 = nn.Linear(latent_dim, latent_dim) 
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(latent_dim, 1)
      self.sigmoid = nn.Sigmoid() 

    def forward(self, x): 
      x = self.relu(self.fc1(x)) 
      x = self.relu(self.fc3(x)) 
      x = self.relu(self.fc3(x)) 
      return self.sigmoid(self.fc2(x))

    def toggle_grad(self, enable): 
      for param in self.parameters(): 
        param.requires_grad = enable

