from .one_dim import * 
from .vision import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler 
from typing import Tuple


def init_dataset(cfg_dataset: dict) -> Tuple[Dataset, Dataset, Dataset]:  
  """
  Returns the train/val/test split of the dataset we want to work with.
  """
  match cfg_dataset["name"]: 
    case "mnist": 
      print("Loading dataset: MNIST")
      train_ds, val_ds, test_ds = mnist(cfg_dataset) 
    case "cifar10":
      print("Loading dataset: CIFAR10")
      train_ds, val_ds, test_ds = cifar10(cfg_dataset) 
    case "svhn": 
      print("Loading dataset: SVHN")
      train_ds, val_ds, test_ds = svhn(cfg_dataset) 
    case "celeba": 
      print("Loading dataset: CelebA")
      train_ds, val_ds, test_ds = celebA(cfg_dataset) 
    case "flowers102": 
      print("Loading dataset: Flowers102")
      train_ds, val_ds, test_ds = flowers102(cfg_dataset)
    case "mnist_flat": 
      print("Loading dataset: MNIST Flat")
      train_ds, val_ds, test_ds = mnist_flat(cfg_dataset) 
    case _: 
      raise Exception("Not a valid dataset.")

  # subsample 
  train_samples = cfg_dataset["train"]["subsample"]
  val_samples = cfg_dataset["val"]["subsample"] 
  test_samples = cfg_dataset["test"]["subsample"]
  if isinstance(train_samples, float): 
    train_samples = int(train_samples * len(train_ds))
  if isinstance(val_samples, float): 
    val_samples = int(val_samples * len(val_ds))
  if isinstance(test_samples, float): 
    test_samples = int(test_samples * len(test_ds))
  train_ds = Subset(train_ds, range(train_samples))
  val_ds = Subset(val_ds, range(val_samples))
  test_ds = Subset(test_ds, range(test_samples)) 

  return train_ds, val_ds, test_ds

def init_dataloader(cfg_dataset: dict) -> Tuple[DataLoader, DataLoader, DataLoader]: 
  train_ds, val_ds, test_ds = init_dataset(cfg_dataset)

  if cfg_dataset["is_distributed"]: 
    if cfg_dataset["train"]["shuffle"] or cfg_dataset["val"]["shuffle"] or cfg_dataset["test"]["shuffle"]: 
      UserWarning("You have turned shuffle on when using distributed. Shuffle will be forced off.")
      cfg_dataset["train"]["shuffle"] = False
      cfg_dataset["val"]["shuffle"] = False
      cfg_dataset["test"]["shuffle"] = False

    train_dl = DataLoader(train_ds, 
                          batch_size=cfg_dataset["train"]["batch_size"], 
                          shuffle=cfg_dataset["train"]["shuffle"], 
                          num_workers=cfg_dataset["train"]["num_workers"],
                          sampler=DistributedSampler(train_ds, seed=cfg_dataset["seed"]))
    val_dl = DataLoader(val_ds, 
                        batch_size=cfg_dataset["val"]["batch_size"], 
                        shuffle=cfg_dataset["val"]["shuffle"], 
                        num_workers=cfg_dataset["val"]["num_workers"],
                        sampler=DistributedSampler(val_ds, seed=cfg_dataset["seed"]))
    test_dl = DataLoader(test_ds, 
                         batch_size=cfg_dataset["test"]["batch_size"], 
                         shuffle=cfg_dataset["test"]["shuffle"], 
                         num_workers=cfg_dataset["test"]["num_workers"],
                         sampler=DistributedSampler(test_ds, seed=cfg_dataset["seed"]))
  else: 
    train_dl = DataLoader(train_ds, 
                          batch_size=cfg_dataset["train"]["batch_size"], 
                          shuffle=cfg_dataset["train"]["shuffle"], 
                          num_workers=cfg_dataset["train"]["num_workers"])
    val_dl = DataLoader(val_ds, 
                        batch_size=cfg_dataset["val"]["batch_size"], 
                        shuffle=cfg_dataset["val"]["shuffle"], 
                        num_workers=cfg_dataset["val"]["num_workers"])
    test_dl = DataLoader(test_ds, 
                         batch_size=cfg_dataset["test"]["batch_size"], 
                         shuffle=cfg_dataset["test"]["shuffle"], 
                         num_workers=cfg_dataset["test"]["num_workers"])

  return train_dl, val_dl, test_dl
