from torch import  nn


class Encoder_Embedding(nn.Module):
    def __init__(self,
                 input_size, hidden_size
                 ):
        super(Encoder_Embedding, self).__init__()
        self.enc = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):  # 输入input为  b x input_size x m
        out_put = self.enc(input)
        return out_put