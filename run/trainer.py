import torch
from torch.nn import Module 
from torch.utils.data import DataLoader 
from torch.optim import Optimizer
from tqdm import tqdm
from typing import Dict
from .utils import *  
from copy import deepcopy
import sys
from.model import GAN

class Trainer: 
  """
  Wrapper class that unifies the dataloader, model, loss, and optimizer. 
  Supports the train, val, and test functions which perform 1 epoch and 
  returns a dict of metrics to be logged. 
  """

  def __init__(self, 
               train_dl: DataLoader, 
               val_dl: DataLoader, 
               test_dl: DataLoader, 
               model: GAN, 
               loss: Module, 
               optimizer: Optimizer, 
               device = None): 
    self.train_dl = train_dl
    self.val_dl = val_dl
    self.test_dl = test_dl

    self.model = model

    self.loss = loss
    self.optimizer = optimizer
    self.device = device

    # the metrics to keep track of
    self.metrics = {
      "discriminator_loss" : 0.0, 
      "generator_loss" : 0.0
    }

  @torch.enable_grad 
  def train(self, initial=False) -> Dict:
    metrics = deepcopy(self.metrics)
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)

    for i, (x_true, _) in enumerate(tqdm(self.train_dl, leave=False, file=tqdm_file)): 
      # Set up gradients based on the current step (Discriminator vs Generator)
      if i % 50 != 0: 
        # train discriminator
        self.model.discriminator.toggle_grad(True)
        self.model.generator.toggle_grad(False)
        # sample minibatch from the true data generating distribution
        x_true = x_true.to(self.device) 
        # sample minibatch from the generator
        minibatch_size = x_true.size(0)
        z_gen = self.model.generator.sample(minibatch_size) 
        x_gen = self.model.generator(z_gen).to(self.device) 

        should_be_true = self.model.discriminator(x_true) 
        should_be_false = self.model.discriminator(x_gen)

        loss = - torch.mean(torch.log(should_be_true) + torch.log(1 - should_be_false))  
        metrics["discriminator_loss"] += loss.item()

        if not initial: 
          self.model.discriminator.zero_grad()
          loss.backward()
          self.optimizer.step()
      else: 
        # train generator 
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(True)
        # sample minibatch from the true data generating distribution
        x_true = x_true.to(self.device) 
        # sample minibatch from the generator
        minibatch_size = x_true.size(0)
        z_gen = self.model.generator.sample(minibatch_size) 
        x_gen = self.model.generator(z_gen).to(self.device) 

        should_be_false = self.model.discriminator(x_gen)
        loss = torch.mean(torch.log(1 - should_be_false))
        metrics["generator_loss"] += loss.item()
        if not initial: 
          self.model.generator.zero_grad()
          loss.backward()
          self.optimizer.step()

    return metrics

  def val(self) -> Dict:
    metrics = deepcopy(self.metrics)
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)

    for i, (x_true, _) in enumerate(tqdm(self.val_dl, leave=False, file=tqdm_file)): 
      # sample minibatch from the true data generating distribution
      x_true = x_true.to(self.device) 
      # sample minibatch from the generator
      minibatch_size = x_true.size(0)
      z_gen = self.model.generator.sample(minibatch_size) 
      x_gen = self.model.generator(z_gen).to(self.device) 

      if i % 2 != 0: 
        # train discriminator
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_true = self.model.discriminator(x_true) 
        should_be_false = self.model.discriminator(x_gen)

        loss = - torch.mean(torch.log(should_be_true) + torch.log(1 - should_be_false))  
        metrics["discriminator_loss"] += loss.item()

      else: 
        # train generator 
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_false = self.model.discriminator(x_gen)
        loss = torch.mean(torch.log(1 - should_be_false))
        metrics["generator_loss"] += loss.item()
    return metrics

  def test(self) -> Dict: 
    metrics = deepcopy(self.metrics)
    # Get the original terminal stdout if logger has redirected it
    tqdm_file = getattr(sys.stdout, 'terminal', sys.stdout)

    for i, (x_true, _) in enumerate(tqdm(self.test_dl, leave=False, file=tqdm_file)): 
      # sample minibatch from the true data generating distribution
      x_true = x_true.to(self.device) 
      # sample minibatch from the generator
      minibatch_size = x_true.size(0)
      z_gen = self.model.generator.sample(minibatch_size) 
      x_gen = self.model.generator(z_gen).to(self.device) 

      if i % 2 != 0: 
        # train discriminator
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_true = self.model.discriminator(x_true) 
        should_be_false = self.model.discriminator(x_gen)

        loss = - torch.mean(torch.log(should_be_true) + torch.log(1 - should_be_false))  
        metrics["discriminator_loss"] += loss.item()

      else: 
        # train generator 
        self.model.discriminator.toggle_grad(False)
        self.model.generator.toggle_grad(False)

        should_be_false = self.model.discriminator(x_gen)
        loss = torch.mean(torch.log(1 - should_be_false))
        metrics["generator_loss"] += loss.item()
    return metrics

