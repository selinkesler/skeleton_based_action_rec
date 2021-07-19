import torch
import torch.nn as nn
import numpy as np
   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
class GRU(nn.Module) :

    def __init__(self, n_class=61, input_size=30, n_layers = 2, drop_prob = 0, n_hidden = 256, batch_size = 256):
    
        super(GRU, self).__init__()
        
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.batch_size = batch_size

        
        # input_size =  30 = 15 * 2 because of x and y coordinates for each joint
        self.gru = nn.GRU(input_size, n_hidden, n_layers, dropout=drop_prob, batch_first = True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, n_class)
        
        self.sigmoid = nn.Sigmoid()
        
        #self.init_weights()
        
    def forward(self, x, hidden) :
    
        x, hidden = self.gru(x,hidden)
        # print(x.shape)
        # print(x)
        
        
        x = self.dropout(x)
        # print(x)
        
        
        # x = x.view(x.size()[0]*x.size()[1], self.n_hidden) --- not being used, instead squeeze is being used
        
        
        # first taking the last output of LSTM, then using fully connected layer for classification because we only need the information after 32 frames
        x = x.squeeze()[:,-1]
        # print(x.shape)
        # print(x)
        
        
        x = self.fc(x)
        # print(x.shape)
        #print(x)
        
        # w = x.squeeze()[-1,:] -- was not sure where to use it and which form 
        
        
        
        # x = x.view(256, -1)
        # x = x[:,-1]
        # print(x.shape)
        
        return x, hidden
        
    '''    
    def init_hidden_train(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        #.to(device)
        return hidden
    ''' 
        
    def init_hidden_train(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device)
        return hidden    
        
        
        
    '''    
    def init_hidden_test(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        #.to(device)
        return hidden
	'''

	
	
    '''
    def forward(self, x, hc) :
    
        x, (h, c) = self.lstm(x,hc)
        
        x = self.dropout(x)
        
        x = x.view(x.size()[0]*x.size()[1], self.n_hidden)
        
        x = self.fc(x)
        
        return x, (h, c)
        
    ''' 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(ShuffleNetV2, self).__init__()

        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)
        
        # self.conv_51 = conv_bn(24, 24, 2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size / 32)))

        # building classifier
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        
        
        
        # for refrence joint purposes 
        # x = self.conv_51(x)
        # print(x.shape)
        
        x = self.features(x)
        # print(x.shape)
        x = self.conv_last(x)
        # print(x.shape)
        x = self.globalpool(x)
        # print(x.shape)
        x = x.view(-1, self.stage_out_channels[-1])
        # print(x.shape)
        x = self.classifier(x)
        
        # print ("outputs :")
        # print(x)
        # x = x + x
        # print("deneme")
        # print(x)
        
        # print(x.shape)
        return x
        
        

