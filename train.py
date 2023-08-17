import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import torch
import numpy as np

from loguru import logger
from model.drpo import DRPO
from trainer.drpo_trainer import DRPOTrainer
from torch.utils.tensorboard import SummaryWriter

# load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def main(config):
    ''' Device initialization '''
    device = torch.device('cuda')
    config["device"] = device

    ''' Trainer & Model & Optimizer initialization '''
    trainer = DRPOTrainer(config)
    model = DRPO(config).to(device)
    for name, param in model.named_parameters():
        if "weight" in name:
            torch.nn.init.orthogonal_(param)
    optimizer = torch.optim.AdamW( model.parameters(), lr=config["learning_rate"] )

    ''' Start Training '''
    trainer.train(model, optimizer)

if __name__ == "__main__":
    main(config)
