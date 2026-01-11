import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent_Feudal(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent_Feudal, self).__init__()
        self.args = args

        # [修复 1] 安全读取 goal_dim，如果没有则默认为 0 (兼容普通模式)
        self.goal_dim = getattr(args, "goal_dim", 0)
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.hyper_embed_dim = getattr(args, "hypernet_embed", 64)

        # 1. 输入归一化
        self.input_norm = nn.LayerNorm(input_shape)

        # 2. 基础线性层 (Base Layer) - 这是 Agent 的"本体"
        self.fc_base = nn.Linear(input_shape, args.rnn_hidden_dim)

        # 3. Hypernetwork (仅当 goal_dim > 0 时才初始化)
        self.n_weights = input_shape * self.rnn_hidden_dim
        self.n_bias = self.rnn_hidden_dim

        if self.goal_dim > 0:
            self.hyper_net = nn.Sequential(
                nn.Linear(self.goal_dim, self.hyper_embed_dim),
                nn.ReLU(),
                nn.Linear(self.hyper_embed_dim, self.n_weights + self.n_bias)
            )
        else:
            self.hyper_net = None

        self.layer_norm_2 = nn.LayerNorm(args.rnn_hidden_dim)

        self.feat_extract_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        )

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.val_head = nn.Linear(args.rnn_hidden_dim, 1)
        self.adv_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc_base.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # [修复 2] goal 设置为可选参数，兼容 BasicMAC 的调用
    def forward(self, inputs, hidden_state, goal=None):
        bs, input_dim = inputs.shape
        
        # 1. 归一化输入
        inputs = self.input_norm(inputs)

        # 2. 计算基础特征 (Base Features)
        x_base = self.fc_base(inputs) 

        # 3. 计算 Hypernetwork 特征 (仅当 goal 存在时)
        if goal is not None and self.hyper_net is not None:
            hyper_out = self.hyper_net(goal)
            hyper_out = F.tanh(hyper_out) * 0.5 

            weights = hyper_out[:, :self.n_weights].view(bs, input_dim, self.rnn_hidden_dim)
            bias = hyper_out[:, self.n_weights:].view(bs, self.rnn_hidden_dim)
            
            # 动态权重计算
            x_hyper = th.bmm(inputs.unsqueeze(1), weights).squeeze(1) + bias
        else:
            # 如果没有 goal (普通模式)，修正项为 0
            x_hyper = 0

        # 4. 残差连接 (x_base + 0 = x_base，退化为普通 RNN)
        x = x_base + x_hyper
        
        # 后续处理
        x = self.layer_norm_2(x)
        x = self.feat_extract_2(x)
        
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        
        val = self.val_head(h)
        adv = self.adv_head(h)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q, h