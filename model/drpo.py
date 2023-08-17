import torch

class DRPO(torch.nn.Module):
    def __init__(self, config):
        super(DRPO, self).__init__()
        self.device = config["device"]                      # "cuda" 
        self.hidden_size = config["hidden_size"]            # 256
        self.sequence_length = config["sequence_length"]    # 140
        self.feature_size = config["feature_size"]          # 49 * 30
        self.stock_num = config["stock_num"]                # 30
        self.bias = config["bias"]                          # True
        self.dropout = config["dropout_rate"]               # 0.25
        self.lstm_layers = config["lstm_layers"]            # 3
        self.buy_cost_pct = config["buy_cost_pct"]          # 6.87e-05
        self.sell_cost_pct = config["sell_cost_pct"]        # 0.0010687
        self.total_money = config["total_money"]            # 100
        self.gamma = config["gamma"]                        # 1
        self.head_mask = config["head_mask"]                # 10
        self.tail_mask = config["tail_mask"]                # 130
        self.reward_scaling = config["reward_scaling"]      # 1000
        self.config = config
        # self.faltten = torch.nn.Flatten(start_dim=2)
        # self.LSTMs = torch.nn.ModuleList()
        # self.MPLs = torch.nn.ModuleList()
        # for i in range(self.lstm_layers):
        #     if i % self.lstm_layers == 0:
        #         self.LSTMs.append(
        #             torch.nn.LSTMCell(
        #                 input_size=self.feature_size+self.stock_num,
        #                 hidden_size=self.hidden_size,
        #                 bias=self.bias,
        #             )
        #         )
        #     else:
        #         self.LSTMs.append(
        #             torch.nn.LSTMCell(
        #                 input_size=self.hidden_size,
        #                 hidden_size=self.hidden_size,
        #                 bias=self.bias,
        #             )
        #         )

        self.LSTM = torch.nn.LSTM(
            input_size = self.feature_size+1,
            hidden_size = self.hidden_size,
            num_layers = self.lstm_layers,
            bias = self.bias,
            batch_first = True,
            dropout = self.dropout,
        )

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * self.stock_num,
                            self.hidden_size * 7, bias=self.bias),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.hidden_size * 7, 128, bias=self.bias),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(128, 64, bias=self.bias),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(64, 1, bias=self.bias),
        )


    def forward(self, inputs, buy_price_batch, sell_price_batch, difficulty=1):
        ''' Step 0: Initialization'''
        bs, ts, sn, fn = inputs.shape                                       # bs, 140, 30, 49
        mid_price_batch = (buy_price_batch + sell_price_batch) / 2          # bs, 140, 30, 49
        buy_price_batch_ = mid_price_batch + \
            (buy_price_batch - mid_price_batch) * difficulty                # bs, 140, 30, 49
        sell_price_batch_ = mid_price_batch + \
            (sell_price_batch - mid_price_batch) * difficulty               # bs, 140, 30, 49

        self.buy_cost_pct *= difficulty
        self.sell_cost_pct *= difficulty

        out_time_stock = []
        hx_cx_stock = []

        state = torch.zeros(bs, ts+1, sn, device=self.device)               # (bs, 141, 30) holding 
        new_state = torch.zeros(bs, ts+1, sn, device=self.device)           # (bs, 141, 30) holding of the next state
        cash = torch.zeros(bs, ts+1, device=self.device)                    # (bs, 141, 30) cash
        hold = torch.zeros(bs, ts+1, device=self.device)                    # (bs, 141, 30) value of holding
        asset = torch.zeros(bs, ts+1, device=self.device)                   # (bs, 141, 30) total asset

        # Equalizing
        state[:, 0, :] = torch.tensor([100]*self.stock_num, device=self.device).\
            unsqueeze(0).repeat_interleave(bs, dim=0)
        cash[:, 0] = self.total_money                                                # first date cash
        hold[:, 0] = torch.sum( state[:, 0, :] * mid_price_batch[:, 0, :], dim=-1 )  # first date hold
        asset[:, 0] = cash[:, 0] + hold[:, 0]                                        # first date asset

        ''' Step 1: Model Inferencing '''

        hx = torch.ones(self.lstm_layers, bs* sn, self.hidden_size,
                                device=self.device)
        cx = torch.ones(self.lstm_layers, bs* sn, self.hidden_size,
                                device=self.device)
        obs_omega = torch.softmax(state[:, 0, :], dim=-1)
        
        # inputs = self.faltten(inputs)                                       # bs, 140, 1470
        for i in range(ts):  # each timestep
            input = inputs[:, i:i+1, :]                                         # bs, 30, 49
            input = input.reshape(-1, fn)
            
            input_temp = torch.concat([input, obs_omega.reshape(-1,1)], dim=1)
            input_temp = input_temp.reshape(bs* sn, 1, -1)
            output, (hx, cx) = self.LSTM(input_temp, (hx, cx))
            output = output.reshape(bs, -1)
            output = self.MLP(output)
           
            # output = output - torch.mean(output,dim = -1,keepdim=True)
            # output = torch.sigmoid(output)
            out_time_stock.append(output)  # (batchsize,stocknum)

    
            ''' Step 2: Calculate Next State '''
            with torch.no_grad():
                this_state = state[:, i, :]                                     # (bs, 30)   
                output = output - torch.mean(output,dim = -1,keepdim=True)      # (bs, 30)
                next_state = this_state + output                                # change to next state
                # check next hold > 0
                zero_actions = torch.zeros_like(next_state).to(self.device)     # (bs, 30)
                next_state = torch.where(
                    next_state <= 0, zero_actions, next_state)  # <=0 填0       # (bs, 30)
                # check cash > 0
                weight_change = next_state - this_state                         # (bs, 30)
                zero_actions = torch.zeros_like(weight_change).to(self.device)  # (bs, 30)
                sell_actions = torch.where(
                    weight_change >= 0, zero_actions, weight_change)            # (bs, 30)
                buy_actions = torch.where(
                    weight_change <= 0, zero_actions, weight_change)            # (bs, 30)

                buy_costs = -1 * buy_actions * \
                    buy_price_batch_[:, i-1, :]*(self.buy_cost_pct)   # <0      # (bs, 30)
                sell_costs = -1 * sell_actions * \
                    sell_price_batch_[:, i-1, :]*(-self.sell_cost_pct)  # <0    # (bs, 30)

                sell = torch.sum(
                    (-1 * sell_actions * sell_price_batch_[:, i-1, :] + sell_costs), dim=-1)  # (bs, 30) <0
                this_cash = cash[:, i-1]                                        # (bs, 30)
                this_cash = this_cash + sell                                    # (bs, 30)

                buy = torch.sum(
                    (buy_actions * buy_price_batch_[:, i-1, :] - buy_costs), dim=-1)  # (bs, 30) >0
                next_cash = this_cash - buy                                     # (bs, 30)

                # Mandatory use of all cash
                factor = this_cash / buy                                        # (bs, 30)
                temp_factor = torch.zeros_like(factor)                          # (bs, 30)
                factor = torch.where(torch.isnan(factor), temp_factor, factor)  # (bs, 30)
                factor = torch.where(torch.isinf(factor), temp_factor, factor)  # (bs, 30)

                buy_actions = buy_actions * \
                    factor.unsqueeze(-1).repeat_interleave(sn, dim=-1)          # (bs, 30)
                buy = buy * factor                                              # (bs, 30)
                next_cash = this_cash - buy                                     # (bs, 30)
                weight_change = sell_actions + buy_actions                      # (bs, 30)
                next_state = this_state + weight_change                         # (bs, 30)

                this_hold = torch.sum(
                    next_state * mid_price_batch[:, i-1, :], dim=-1)            # (bs, 30)

                # Record the results
                state[:, i+1, :] = next_state                                   # (bs, 30)
                cash[:, i+1] = next_cash                                        # (bs, )
                hold[:, i+1] = this_hold                                        # (bs, )
                asset[:, i+1] = next_cash + this_hold                           # (bs, )

                obs_omega = torch.softmax(next_state, dim=-1)                   # (bs, 30)

        out_time_stock = torch.stack(out_time_stock, dim=1)                     # (bs, 140, 30)

        
        with torch.no_grad():
            new_state[:, :-1, :] = state[:, 1:, :]
            state_change = (new_state-state)[:, :-1, :]
            asset = asset / asset[:, 0].unsqueeze(1).repeat_interleave(ts+1, dim=1)
            new_asset = torch.zeros_like(asset, device=self.device)
            new_asset[:, :-1] = asset[:, 1:]
            rewards = (new_asset-asset)[:, :-1]
            rewards_pct = rewards / asset[:, 0].unsqueeze(1).repeat_interleave(ts, dim=1)

            ''' Step 3: Calculate Expectations by Dynamic Programming '''
            bs, ts, sn = state_change.shape
            E_pi = torch.zeros(bs, ts).to(self.device)
            final_price = asset[:, -1]
            initial_price = asset[:, 0]
            E_pi[:, -1] = final_price / initial_price
            for i in range(ts-2, -1, -1):
                E_pi[:, i] = rewards_pct[:, i] + self.gamma * E_pi[:, i+1]

            ''' Step 4: Calculate Reward '''
            head_mask = self.head_mask
            tail_mask = self.tail_mask

            actual_reward = torch.sum(rewards_pct, dim=-1)
            profit = torch.mean(E_pi[:, 0])
            rewards_pct = rewards_pct[:, head_mask:tail_mask]

            E_pi = E_pi[:, head_mask+1:tail_mask+1]
            V_pi = rewards_pct + self.gamma * E_pi
            V_pi = V_pi * self.reward_scaling
            V_pi = V_pi - torch.mean(V_pi, dim=1, keepdim=True)
            V_pi = V_pi.unsqueeze(-1).repeat(1, 1, out_time_stock.shape[-1])

        ''' Step 5: Calculate Loss '''
        loss = -torch.mean(V_pi * out_time_stock[:, head_mask:tail_mask, :])
        actual_reward = torch.mean(actual_reward)

        return out_time_stock, loss, profit, actual_reward, asset