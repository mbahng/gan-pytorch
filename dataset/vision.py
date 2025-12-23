from torchvision import transforms, datasets 
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.nn as nn

def mnist(cfg_dataset: dict): 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor(), 
    nn.Flatten(start_dim=0)
  ])

  # split
  train_split, val_split, _ = cfg_dataset["split"] 
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True) 

  return train_ds, val_ds, test_ds

def cifar10(cfg_dataset: dict): 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = cfg_dataset["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True) 

 
  return train_ds, val_ds, test_ds

def svhn(cfg_dataset: dict): 
  """
  There is also an "extra" split for SVHN
  """
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = cfg_dataset["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.SVHN(root='./dataset', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.SVHN(root='./dataset', train=False, transform=transform, download=True) 

  return train_ds, val_ds, test_ds

def celebA(cfg_dataset: dict): 
  raise Exception("This is broken. Fix.")
  return datasets.CelebA(
    root='./dataset', 
    split='all',
    transform=transforms.Compose([
      transforms.ToTensor()
    ]), 
    download=True
  )

def flowers102(cfg_dataset: dict):
  """
  Needs scipy to load target files
  """
  transform = transforms.Compose([
    transforms.ToTensor()
  ])


  train_ds = datasets.Flowers102(root='./dataset', split="train", transform=transform, download=True) 
  val_ds = datasets.Flowers102(root='./dataset', split="val", transform=transform, download=True) 
  test_ds = datasets.Flowers102(root='./dataset', split="test", transform=transform, download=True) 

  return train_ds, val_ds, test_ds
