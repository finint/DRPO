import numpy as np
import torch
import yaml
import os
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from model.drpo import DRPO
from trainer.drpo_trainer_distributed import DRPOTrainer
import warnings
warnings.filterwarnings("ignore")

# load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@logger.catch()
def main(config):
    ''' Distributed training initialization '''
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'gloo',
        init_method='env://'
    )
    device = torch.device(f'cuda:{local_rank}')
    config["device"] = device

    ''' Trainer & Model & Optimizer initialization '''
    trainer = DRPOTrainer(config)
    model = DRPO(config).to(device)
    for name, param in model.named_parameters():
        if "weight" in name:
            torch.nn.init.orthogonal_(param)
    model = torch.nn.parallel.DistributedDataParallel( model, device_ids=[local_rank] )  
    optimizer = torch.optim.Adam( model.parameters(), lr=config["learning_rate"] )

    ''' Start Training '''
    model, optimizer = trainer.train(model, optimizer)

if __name__ == "__main__":
    main(config)
