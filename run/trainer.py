import torch
from torch.nn import Module 
from torch.utils.data import DataLoader 
from torch.optim import Optimizer
from tqdm import tqdm
from typing import Dict
from .utils import *  
from copy import deepcopy
import sys

class Trainer: 
  """
  Wrapper class that unifies the dataloader, model, loss, and optimizer. 
  Supports the train, val, and test functions. 
  """

  def __init__(self, 
               train_dl: DataLoader, 
               val_dl: DataLoader, 
               test_dl: DataLoader, 
               model: Module, 
               loss: Module, 
               optimizer: Optimizer, 
               device: int = None): 
    self.train_dl = train_dl
    self.val_dl = val_dl
    self.test_dl = test_dl

    self.model = model

    self.loss = loss
    self.optimizer = optimizer
    self.device = device

    # the metrics to keep track of
    self.metrics = {
      "total_loss" : 0.0, 
      "n_correct" : 0, 
      "n_samples" : 0, 
    }

  @torch.enable_grad
  def train(self, initial=False) -> Dict:
    metrics = deepcopy(self.metrics)
    self.model.train()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for x, y in tqdm(self.train_dl, leave=False, file=tqdm_file):
      x, y = x.to(self.device), y.to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)

      if not initial: 
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
      metrics["total_loss"] += loss.item() 

      # Calculate accuracy
      pred_labels = y_pred.argmax(dim=1)
      correct = (pred_labels == y).sum().item() 
      metrics["n_correct"] += correct
      metrics["n_samples"] += y.size(0)
    return metrics

  @torch.no_grad
  def val(self) -> Dict:
    metrics = deepcopy(self.metrics)
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for x, y in tqdm(self.val_dl, leave=False, file=tqdm_file):
      x, y = x.to(self.device), y.to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)
      metrics["total_loss"] += loss.item()

      # Calculate accuracy
      pred_labels = y_pred.argmax(dim=1)
      correct = (pred_labels == y).sum().item() 
      metrics["n_correct"] += correct
      metrics["n_samples"] += y.size(0)
    return metrics

  @torch.no_grad
  def test(self) -> Dict: 
    metrics = deepcopy(self.metrics)
    self.model.eval()
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)
    for x, y in tqdm(self.test_dl, leave=False, file=tqdm_file):
      x, y = x.to(self.device), y.to(self.device)
      y_pred = self.model(x)
      loss = self.loss(y_pred, y)
      metrics["total_loss"] += loss.item()

      # Calculate accuracy
      pred_labels = y_pred.argmax(dim=1)
      correct = (pred_labels == y).sum().item() 
      metrics["n_correct"] += correct
      metrics["n_samples"] += y.size(0)
    return metrics

