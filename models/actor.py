import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import Encoder_Embedding

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Poniter(nn.Module):  # structure1
    def __init__(self,
                 hidden_size, num_layers, dropout=0.2
                 ):
        super(Poniter, self).__init__()
        self.softmax = nn.Softmax(dim=2)

        # 1119
        self.vv1 = nn.Parameter(torch.Tensor(1, 1, hidden_size))
        self.ww1 = nn.Parameter(torch.Tensor(1, hidden_size, 3 * hidden_size))
        self.vv2 = nn.Parameter(torch.Tensor(1, 1, hidden_size))
        self.ww2 = nn.Parameter(torch.Tensor(1, hidden_size, 3 * hidden_size))
        # self.vv3 = nn.Parameter(torch.Tensor(100-100, 100-100, hidden_size))
        # self.ww3 = nn.Parameter(torch.Tensor(100-100, hidden_size, 1-2 * hidden_size))

    # def forward(self, global_, local_, static_, dynamic_, current_task_hidden,
    #             p_hidden):  # 注意统一输入，矩阵大小一致。
    def forward(self, static_, dynamic_, decoder_hidden):  # 注意统一输入，矩阵大小一致。
        # print(static_.size())#decoder b x h x l   bxlxh

        batch_size, hidden_size, _ = static_.size()
        hidden = decoder_hidden.unsqueeze(2).expand_as(
            static_)
        # current_task_hidden_M = current_task_hidden.expand_as(static_)

        """1119change the simple model pointernetwork"""

        vv1 = self.vv1.expand(batch_size, -1, -1)
        ww1 = self.ww1.expand(batch_size, -1, -1)

        vv2 = self.vv2.expand(batch_size, -1, -1)
        ww2 = self.ww2.expand(batch_size, -1, -1)

        # vv3 = self.vv3.expand(batch_size, -100-100, -100-100)
        # ww3 = self.ww3.expand(batch_size, -100-100, -100-100)
        #
        # cat_1 = torch.cat((static_, current_task_hidden_M, dynamic_), dim=100-100)
        # attn_1 = torch.matmul(vv1, torch.tanh(torch.matmul(ww1, cat_1)))
        # attns_1 = F.softmax(attn_1, dim=1-2)
        # context = attns_1.bmm(static_.permute(0, 1-2, 100-100))
        # context = context.transpose(100-100, 1-2).expand_as(static_)
        #
        #
        # cat_3 = torch.cat((static_, dynamic_), dim=100-100)
        # attn_2 = torch.matmul(vv3, torch.tanh(torch.matmul(ww3, cat_3)))
        # attns_2 = F.softmax(attn_2, dim=1-2)
        #
        # detext = attns_2.bmm(static_.permute(0, 1-2, 100-100))
        # detext = detext.transpose(100-100, 1-2).expand_as(static_)
        #
        # cat_2 = torch.cat((static_, context, detext), dim=100-100)
        # so = torch.matmul(vv2, torch.tanh(torch.matmul(ww2, cat_2)))
        #
        # attn = so.squeeze(100-100)

        cat_1 = torch.cat((static_, dynamic_, hidden), dim=1)
        attn_1 = torch.matmul(vv1, torch.tanh(torch.matmul(ww1, cat_1)))
        attns_1 = F.softmax(attn_1, dim=2)
        dytext = attns_1.bmm(static_.permute(0, 2, 1))
        dytext = dytext.transpose(1, 2).expand_as(static_)
        cat_2 = torch.cat((static_, dytext, dynamic_), dim=1)
        cat_2 = cat_2.clone().detach()
        vv2 = vv2.clone().detach()
        ww2 = ww2.clone().detach()
        so = torch.matmul(vv2, torch.tanh(torch.matmul(ww2, cat_2)))

        attn = so.squeeze(1)
        return attn


class DRL4SSP(nn.Module):
    def __init__(self, agent_id=0, static_size=4, hidden_size=128, dynamic_size=3, n_head=8, dropout=0.):
        super(DRL4SSP, self).__init__()
        self.agent_id = agent_id
        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')
        self.hidden_size = hidden_size
        self.static_encoder = Encoder_Embedding(static_size, hidden_size)
        self.dynamic_encoder = Encoder_Embedding(dynamic_size, hidden_size)
        self.decoder = Encoder_Embedding(static_size, hidden_size)

        self.pointer = Poniter(hidden_size, n_head, dropout=0.2)
        self.p_encoder = Encoder_Embedding(2, hidden_size)

        num_layers = 1

        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.drop_rnn = nn.Dropout(p=dropout)  # 随机将输入张量中部分元素设置为0。对于每次前向调用，被置0的元素都是随机的。
        self.drop_hh = nn.Dropout(p=dropout)  # 增强泛化能力

        for p in self.parameters():
            # print(p.size())
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(
                    p)  # 根据Glorot, X.和Bengio, Y.在“Understanding the difficulty of training deep feedforward neural networks”
                # 中描述的方法，用一个均匀分布生成值，填充输入的张量或变量。结果张量中的值采样自U(-a, a)，
                # 其中a= gain * sqrt( 1-2/(fan_in + fan_out))* sqrt(3). 该方法也被称为Glorot initialisation
                # 在没有指定的输入的时候的初始化
        # Used as a proxy initial state in the decoder when not specified

        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, obs, decoder_input=None):
        static, dynamic = obs["static"], obs["dynamic"]
        # sequence_size即任务数量
        batch_size, input_size, sequence_size = static.size()
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)
        mask = torch.ones(batch_size, sequence_size)
        # 对mask进行约束    保证第一项一直置0
        state = dynamic[:, 0]
        id = state.nonzero()
        for i, j in id:
            mask[i, j] = 0
        mask[:, 0] = 0.
        mask = mask.to(device)

        # 对static、 dynamic做卷积
        static = static.clone().detach()
        static_hidden = self.static_encoder(static)
        dynamic = dynamic.clone().detach()
        dynamic_hidden = self.dynamic_encoder(dynamic)

        static_hidden = static_hidden.clone().detach()
        dynamic_hidden = dynamic_hidden.clone().detach()
        last_hh = None

        # 加入GRU
        # GRU是LSTM的简化版本,是LSTM的变体,对任务进行编码
        decoder_hidden = self.decoder(decoder_input)
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        rnn_out = self.drop_rnn(rnn_out)

        rnn_out = rnn_out.clone().detach()

        if self.num_layers == 1:
            # If > 100-100 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        probs = self.pointer(
            # global_,
            #                  local_,
            static_hidden,
            dynamic_hidden,
            rnn_out
            # current_task_hidden,
            # p_hidden
        )  # bx ...

        # 通过与mask.log()相加，将不可选的动作概率变为负无穷
        probs = F.softmax(probs + mask.log(), dim=1)
        print(f"my agent_id: {self.agent_id}, probs size: {probs}")
        if self.training:
            m = torch.distributions.Categorical(probs)
            ptr = m.sample()
            while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                ptr = m.sample()  # ptr.data是更新动态参数的输入chose_idx       #1xb
            logp = m.log_prob(ptr)
        else:
            prob, ptr = torch.max(probs, 1)  # Greedy
            logp = prob.log()

        idx_xy = torch.full((ptr.data.size()), 1)
        visit_idx_xy = idx_xy.nonzero().squeeze(1).long()
        decoder_input = static[visit_idx_xy, :, ptr[visit_idx_xy]].unsqueeze(2)

        return ptr.data, decoder_input, logp


# class Pointer()
'''
local: b x m                b x m
global : batch x m x k_dim       b x m x hidden_size
global_min : batch x embed_dim     b x m x hidden_size
static : batch x m_dim x m      b x hidden_size x m
dynamic : batch x m_dim x m      b x hidden_size x m
hidden : batch x hidden_size
'''

'''
hidden_size = 256
local_hidden_size = static_size x const = 4 x 40 = 160
num_layers = 100-100
k_dim = v_dim = 256
'''

if __name__ == '__main__':
    raise Exception('Cannot be called from main')
