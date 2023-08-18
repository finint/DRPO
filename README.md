# Efficient Continuous Space Policy Optimization for High-frequency Trading
## 1. Prepare you training data

* The input of our model are using a .npz file which contains `input_data, buy_price, sell_price` and `date_list` by default. Of course, you may use the data format you wish by changing the `_load_data` function of the `DRPOTrainer` class in `./trainer/drpo_trainer.py`. But to make sure that the `input_date`'s shape must be `(batchsize, sequence_length, stock_num, feature_num)` and the `buy_price`'s and `sell_price`'s shape must be `(batchsize, sequence_length, stock_num)` accordingly. `Date_list` is not necessarily needed in training.

## 2. Train you model

* Before training, make sure to change the parameters in `./config/config.yaml`

  ``` yaml
  batch_size: 128            # batchsize
  bias: true                 # whether to use bias 
  hmax: 100                  # scaling actions, which is good for training. Only used in 'multiple' model type now 
  buy_cost_pct: 6.87e-05     # may change according to your market
  data_path: ./data/SSE50_sample.npz  # your training data path. shape: (batchsize, sequence_length, stock_num, feature_num)
  dropout_rate: 0.25         # drpoout rate of the whole network
  feature_size: 1470         # feature_num   49 for SSE50, 6 for DOW30, 43 for COIN
  gamma: 1                   # gamma
  head_mask: 10              # get rid of the fisrt 10 timesteps since the historical data are not sufficient 
  hidden_size: 50            # hidden_size of the LSTM
  learning_rate: 0.0001      # learning rate of the model
  lstm_layers: 3             # layers of the LSTM network
  model_path: ./saved_models # model save path
  reward_scaling: 1000       # good for training
  sell_cost_pct: 0.0010687   # may change according to your market
  sequence_length: 140       # the timestep of each bash
  stock_num: 30              # stock numbers   30 for SSE50, 30 for DOW30, 3 for COIN
  tail_mask: 130             # get rid of the last 10 timesteps since the expectation results may not be credible enough
  tb_path: ./tb_results/     # tensorborad log path 
  total_money: 100           # inital money
  model_save_interval: 5     # interval of saving the model
  use_difficulty: True       # Whether to use difficulty (Good for training)
  total_steps: 100           # In these steps difficulty are changing gradually from 0 to 1
  epochs_per_step: 10        # epochs in each step
  ```

* Install required packages

  ``` shell
  pip install -r requirements.txt  # requirements_full.txt for specific versions
  ```

* Training (Single GPU)

  ``` shell
  sh train.sh
  ```

* Training (Parallel)

  ``` shell
  sh train_parallel.sh
  ```

## 3. Citing

* If you find this papaer is useful for your research, please consider citing the following papers:

  ``` latex
  @inproceedings{han2023efficient,
      title={Efficient Continuous Space Policy Optimization for High-frequency Trading},
      author={Han, Li and Ding, Nan and Wang, Guoxuan and Cheng, Dawei and Liang, Yuqi},
      booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
      pages={4112--4122},
      year={2023}
    }
  ```

