import os
import sys
import yaml
import torch
from torch.nn import Module
from torch.optim import Optimizer
from ..trainer import Trainer
import wandb
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict

class Logger: 
  """
  Logs and saves the following. 
    - configuration files 
    - Metrics as json 
    - Model and optimize state_dicts 
    - figures and visuals per epoch
    - logs to wandb 
  Should add custom functions for saving other data specific to your project. 
  """

  def __init__(self, cfg, device=None):
    cfg_log = cfg["log"]
    self.savedir = cfg_log["savedir"]
    self.save_model = cfg_log["save_model"]
    self.save_optimizer = cfg_log["save_optimizer"]

    self.save_every = int(cfg_log["save_every"])
    self.wandb_enabled = cfg_log["wandb"]["enabled"]
    self.is_distributed = True if cfg["n_gpus"] > 1 else False
    self.device = device

    self._makedir()
    self._save_cfg(cfg)
    self._setup_stdout_logging()

    # Only initialize wandb on device 0 to avoid creating multiple runs
    if self.wandb_enabled and (not self.is_distributed or self.device == 0):
      self.wandb_logger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=cfg_log["wandb"]["entity"],
        # Set the wandb project where this run will be logged.
        project=cfg_log["wandb"]["project"],
        # Track hyperparameters and run metadata.
        config=cfg,
        name=cfg["name"],
        # Store wandb data in the same directory as other run data
        dir=self.savedir
      )
    else:
      self.wandb_logger = None
    
  def _makedir(self): 
    os.makedirs(self.savedir, exist_ok=True)

  def _save_cfg(self, cfg):
    cfg_path = os.path.join(self.savedir, "cfg.yml")
    with open(cfg_path, "w") as f:
      yaml.dump(cfg._cfg, f, default_flow_style=False)

  def _setup_stdout_logging(self):
    """Redirect stdout and stderr to both console and stdout.txt"""
    stdout_path = os.path.join(self.savedir, "stdout.txt")
    self.stdout_file = open(stdout_path, 'a+')
    self.terminal = sys.stdout
    self.terminal_err = sys.stderr
    self._at_line_start = True  # Track if we're at the beginning of a line
    sys.stdout = self
    sys.stderr = self

  def write(self, message):
    """Required method for stdout redirection - called by print()"""
    if not message:
      return
    if not self._is_main_process(): 
      return

    device = f"GPU {self.device}" if self.device is not None else "CPU"

    # Process the message character by character to add prefix at line starts
    output = []
    for char in message:
      if self._at_line_start and char not in ('\n', '\r'):
        output.append(f"[{device}] ")
        self._at_line_start = False
      output.append(char)
      if char == '\n':
        self._at_line_start = True

    prefixed_message = ''.join(output)
    self.terminal.write(prefixed_message)
    self.stdout_file.write(prefixed_message)
    self.stdout_file.flush()

  def flush(self):
    """Required method for stdout redirection"""
    self.terminal.flush()
    self.stdout_file.flush()

  def isatty(self):
    """Required method for stdout/stderr redirection - checks if underlying stream is a TTY"""
    return self.terminal.isatty()

  def log(self, message):
    """Explicitly log a message to both console and file"""
    self.write(message)

  def _save_model(self, model: Module|DDP, epoch: int): 
    if isinstance(model, DDP): 
      torch.save(model.module.state_dict(), os.path.join(self.savedir, str(epoch).zfill(4), "model.pt"))
    else: 
      torch.save(model.state_dict(), os.path.join(self.savedir, str(epoch).zfill(4), "model.pt"))

  def _save_optimizer(self, optimizer: Optimizer, epoch: int): 
    torch.save(optimizer.state_dict(), os.path.join(self.savedir, str(epoch).zfill(4), "optimizer.pt"))

  def _is_main_process(self): 
    if self.is_distributed and self.device != 0: 
      return False 
    return True

  def save_state(self, trainer: Trainer, epoch: int): 
    """
    Main saving function for saving models and optimizers
    """
    if not self._is_main_process(): 
      return
    if epoch % self.save_every != 0: 
      return 
    os.makedirs(os.path.join(self.savedir, str(epoch).zfill(4)), exist_ok=True)
    if self.save_model:
      self._save_model(trainer.model, epoch) 
    if self.save_optimizer: 
      self._save_optimizer(trainer.optimizer, epoch) 

  def save_metrics(self, metrics: Dict, epoch: int): 
    """
    Main saving function for saving metrics 
    """
    if not self._is_main_process(): 
      return
    os.makedirs(os.path.join(self.savedir, str(epoch).zfill(4)), exist_ok=True)
    with open(os.path.join(self.savedir, str(epoch).zfill(4), "metrics.json"), 'w') as f:
      json.dump(metrics, f, indent=2)

    if self.wandb_enabled and self.wandb_logger is not None:
      self.wandb_logger.log(metrics, step=epoch)


