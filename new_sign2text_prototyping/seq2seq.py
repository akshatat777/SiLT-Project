from torch import nn
import torch

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.ehidden = 64
        self.elayer = 2
        self.encoder = nn.LSTM(27,self.ehidden,self.elayer)
        self.decoder1 = nn.LSTMCell(64,64)
        self.decoder2 = nn.LSTMCell(64,64)
        self.stopper = nn.Linear(64,1)
        self.classifier = nn.Linear(64,27)
    
    def forward(self, input_seq):
        h = torch.zeros((self.elayer,input_seq.shape[1],self.ehidden))
        c = torch.zeros((self.elayer,input_seq.shape[1],self.ehidden))
        encode_seq = self.encoder(input_seq,(h,c))
        input = encode_seq[-1]
        h1 = torch.zeros((self.elayer,input_seq.shape[1],self.ehidden))
        c1 = torch.zeros((self.elayer,input_seq.shape[1],self.ehidden))
        h2 = torch.zeros((self.elayer,input_seq.shape[1],self.ehidden))
        c2 = torch.zeros((self.elayer,input_seq.shape[1],self.ehidden))
        for i in range(len(input_seq)):
            h1,c1 = self.decoder1(input,(h1,c1))
            h2,c2 = self.decoder2(h1,(h2,c2))
            
