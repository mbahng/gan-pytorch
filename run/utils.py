import torch

def one_hot(labels, num_classes):
  """
  Convert labels to one-hot encoding
  
  Args:
    labels: tensor of shape (batch_size,) with integer class labels
    num_classes: total number of classes
  
  Returns:
    one_hot tensor of shape (batch_size, num_classes)
  """
  return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
