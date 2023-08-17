import os
import torch
import datetime
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

class DRPOTrainer():
    def __init__(self, config):
        self.config = config
        self.global_steps = 0
        self.global_epochs = 0
        self.difficulty = 1
        self.device = config['device']
        self._load_data()
        self._init_tb()

    def _load_data(self):
        dataset_npz = np.load(self.config["data_path"], allow_pickle=True)
        input_data = torch.tensor(dataset_npz["input_data"], dtype=torch.float)
        buy_price = torch.tensor(dataset_npz["buy_price"], dtype=torch.float)
        sell_price = torch.tensor(dataset_npz["sell_price"], dtype=torch.float)

        out_str = f"\nInput_data's shape: {input_data.shape}\nBuy_price's shape: {buy_price.shape}\nSell_price's shape: {sell_price.shape}"
        logger.info(out_str)
        dataset = torch.utils.data.TensorDataset(input_data, buy_price, sell_price)
        self.data_loader = torch.utils.data.DataLoader( dataset=dataset, batch_size=self.config["batch_size"])

    def _init_tb(self):
        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        device_name = self.device
        tb_folder = self.config["tb_path"] + f"{time}_{device_name}/"

        os.makedirs(tb_folder,exist_ok = True)
        self.writer = SummaryWriter(tb_folder)

    def train_one_epoch(self,model,optimizer):
        """ Training the model for one epoch.

        Args:
            model: Model to be trained
            optimizer: The optimizer of this model
        
        Returns:
            model: Trained model
            optimizer: The optimizer of this model
        """

        model.train()
        sum_asset = 0
        sum_loss = 0
        sum_reward = 0
        sum_actual_reward = 0
        data_amount = 0

        for batch_id, data_batch in enumerate(self.data_loader):
            state_batch, ask_price_batch, bid_price_batch = data_batch
            state_batch = state_batch.to(self.device)

            ask_price_batch = ask_price_batch.to(self.device)
            bid_price_batch = bid_price_batch.to(self.device)
            profit_pred, loss, reward, actual_reward, asset = model(
                state_batch, ask_price_batch, bid_price_batch, self.difficulty)

            asset = torch.mean(asset, dim=0)[-1]
            self.writer.add_scalar(
                "training/Step Loss", loss, self.global_steps)

            sum_asset += asset.tolist()
            sum_loss += loss.tolist()
            sum_reward += reward.tolist()
            sum_actual_reward += actual_reward.tolist()
            data_amount += state_batch.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.global_steps += 1

        sum_asset /= (batch_id+1)
        sum_loss /= (batch_id+1)
        sum_reward /= (batch_id+1)
        sum_actual_reward /= (batch_id+1)

        self.writer.add_scalar(
            "training/Epoch Loss", sum_loss, self.global_epochs)

        self.global_epochs += 1

        if self.global_epochs % self.config["model_save_interval"] == 0:
            # model_save_path
            model_save_folder = "%s/[difficulty = %.3f]_%02d_%s__%s" % (
                self.config["model_path"],
                self.difficulty,
                self.global_epochs,
                str(self.device),
                datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S_%f')
            )
            os.makedirs(model_save_folder, exist_ok=True)
            save_path = "%s/evolve_%02d_difficulty_%.6f.pth" % (
                model_save_folder, self.global_epochs, self.difficulty)
            torch.save(model.state_dict(), save_path)
        
        return model, optimizer

    def train(self,model,optimizer):
        """ Progressive training

        Training the model with progressive difficulty. 

        Args:
            model: Model to be trained
            optimizer: The optimizer of this model

        """
        
        total_steps = self.config["total_steps"]
        epochs_per_step = self.config["epochs_per_step"]
        
        for it in tqdm(range(total_steps+1)):
            if self.config["use_difficulty"]:
                self.difficulty = it/total_steps

            for epoch_id in range(epochs_per_step):
                model,optimizer = self.train_one_epoch(model,optimizer)
        
        logger.info('\nTraining finished.')
