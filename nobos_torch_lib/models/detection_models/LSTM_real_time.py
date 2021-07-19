import torch
import torch.nn as nn
import numpy as np


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
    
    

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


        
        
        
        
class LSTM(nn.Module) :

    def __init__(self, n_class=61, input_size=30, n_layers = 2, drop_prob = 0, n_hidden = 256, batch_size = 256):
    
        super(LSTM, self).__init__()
        
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.batch_size = batch_size

        
        # input_size =  30 = 15 * 2 because of x and y coordinates for each joint
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, dropout=drop_prob, batch_first = True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, n_class)
        
        self.sigmoid = nn.Sigmoid()
        
        #self.init_weights()
        
    def forward(self, x, hidden) :
    
        x, hidden = self.lstm(x,hidden)
        # print("shapes :")
        # print(x.shape)
        # print(x)
        
        
        x = self.dropout(x)
        # print(x)
        
        
        # x = x.view(x.size()[0]*x.size()[1], self.n_hidden) --- not being used, instead squeeze is being used
        
        # first taking the last output of LSTM, then using fully connected layer for classification because we only need the information after 32 frames
        tensor = torch.cuda.FloatTensor(1, 256).fill_(0)
        for i in range(256):
            tensor[0][i] = x[0][31][i]  
                
        x = self.fc(tensor)
        '''
		else:
            x = x.squeeze()[:,-1]
            x = self.fc(x)
        
        '''
        # NORMALDE KULLANILAN (1 BATCH SIZE DISINDA)
        # x = x.squeeze()[:,-1]

        # print(x.shape)
        # print(x)
        
        # w = x.squeeze()[-1,:] -- was not sure where to use it and which form 
        # x = x.view(256, -1)
        # x = x[:,-1]
        
        return x, hidden
        
        
    def init_hidden_train(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        #.to(device)
        return hidden
        
    def init_hidden_test(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        #.to(device)
        return hidden

    '''
    def forward(self, x, hc) :
    
        x, (h, c) = self.lstm(x,hc)
        
        x = self.dropout(x)
        
        x = x.view(x.size()[0]*x.size()[1], self.n_hidden)
        
        x = self.fc(x)
        
        return x, (h, c)
        
    ''' 
        
        
        
        

