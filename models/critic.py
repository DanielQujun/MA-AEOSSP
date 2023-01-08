import torch
import torch.nn.functional as F
import torch.nn as nn
from models.utils import Encoder_Embedding


class StateCritic(nn.Module):
    # 定义Critic网络
    # 评价网络的复杂度
    """
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, agent_num=1, static_size=4, dynamic_size=3, hidden_size=128):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder_Embedding(static_size * agent_num, hidden_size)  # fliter2 x 1114  128个
        self.dynamic_encoder = Encoder_Embedding(dynamic_size * agent_num, hidden_size)  # fliter1 x 1114  128个
        self.p_encoder = Encoder_Embedding(2, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)  # fliter256 x 1114  20个
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)  # fliter20 x 1114   20个
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)  # fliter20 x 1114    1个

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        static_hidden = self.static_encoder(static)  # 128维向量
        dynamic_hidden = self.dynamic_encoder(dynamic)  # 128维向量

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)  # 横向连接到一起      [s d]

        output = F.relu(self.fc1(hidden))  # relu（20维向量
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output
