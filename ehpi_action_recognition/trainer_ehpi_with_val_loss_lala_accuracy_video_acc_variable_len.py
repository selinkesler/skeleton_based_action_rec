from typing import Dict, List

import numpy as np
import torch
from nobos_commons.tools.log_handler import logger
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class TrainerEhpi(object):
    def train(self, train_loader: DataLoader, train_config: TrainingConfigBase, model, test_loader: DataLoader = None):
        print("Train model: {}".format(train_config.model_name))

        model.to('cuda')

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        #  momentum=train_config.momentum,  momentum=train_config.momentum,   -- only for SGD not used in Adam 

        losses_out = []
        losses_out_test =[]
        
        accuracies_out = []
        accuracies_out_seq =[]

        tb = SummaryWriter(comment=f' batch_size={train_config.batch_size} lr={train_config.learning_rate}')
        
        h_1 = model.init_hidden_train(256)
        h_2 = model.init_hidden_train(1)
        
        
        for epoch in range(train_config.num_epochs):
        
            losses = []
            losses_test = []
            
            model.train()
            print("training model")
            
            train_config.learning_rate_scheduler(optimizer, epoch)
                      
            for i, data in enumerate(train_loader):
            
                x = Variable(data["x"]).to(device)
                y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
                seq = Variable(torch.tensor(data["seq"], dtype=torch.long)).to(device)
                
                # print("input x to model : (in TrainerEhpi)")
                # print(x)
                # print("output y")
                # print(y)
                # print("sequenz ")
                # print(seq)
                
                
                
                # print(x.shape)
                
                x_ = x.cpu().numpy()
                
                # print(x_[2])
                # print(y[2])
                # print(seq[2])
                
                
                
                
                # a = np.zeros((256, 2, 32, 15))
                # a = np.zeros((256, 2, 64, 15))
                a = np.zeros((256, 2, 32, 15))
                
                a[:,0,:,:] = x_[:,0,:,:]
                a[:,1,:,:] = x_[:,1,:,:]
              

                
                data_new_3 = np.transpose(a, (0,2,1,3))
                # print("dimension denemeleri new data_3")
                # print(data_new_3)
                # print(data_new_3.shape)
                
                # input_LSTM = data_new_3.reshape(256,32,30)
                # input_LSTM = data_new_3.reshape(256,64,30)
                input_LSTM = data_new_3.reshape(256,32,30)
                
                # print("dimension denemeleri new data_4")
                # print(data_new_4)
                # print(data_new_4.shape)
                
                
                input_LSTM = torch.Tensor(input_LSTM).to(device)
                # input_LSTM = Variable(data["input_LSTM"]).to(device)
                
                # x = np.zeros((256, 2, 15))
                
                
                print("index_train : " +str(i))
                
                # FOR LSTM 
                h_1 = tuple([e.data for e in h_1])
                
                
                # FOR GRU
                # h_1 = h_1.data
                
                batch = 256
                optimizer.zero_grad()
                outputs,hiden = model(input_LSTM, h_1,batch,32)
                
                # print("outputs (TrainerEhpi):")
                # print(outputs)
                
                loss = loss_func(outputs, y)
                loss.backward()
                losses.append(loss.item())
                
                # print("losses_append")
                # print(losses)
                # print(len(losses))
                optimizer.step()

                
            # loss average overall the batches will be calculated
            loss_total = sum(losses) / len(losses)
            losses_out.append(loss_total)
            
            print("TRAINING LOSS :")
            print("{}: {}".format(epoch, loss_total))
            '''
            # for validation loss in each epoch : 
            with torch.no_grad():
                model.eval()
                print("validation model")   
                
                for i,data in enumerate(test_loader):
                    x = Variable(data["x"]).to(device)
                    y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
                    lala = data["y"].numpy()
                    seq = Variable(torch.tensor(data["seq"], dtype=torch.long)).to(device)
                    # print("seq")
                    # print(seq)
                    # print(seq[1])
                    
                    
                    
                    # print("input x to model : (in TEST)")
                    # print(x)
                    # print("output y")
                    # print(y)                   
                    x_ = x.cpu().numpy()
                   # print("X NEYMIS BAKALIM")
                   # print(x_)
                   # print(x_.shape)
                    
                    
                    
                    
                    original_len = int(x_[0,13500])
                    # original_len = x_[-1]
                    # print("original_len")
                    # print(original_len)
                    
                    new_x = np.zeros((1,original_len))
                    
                
                    a = 0
                    for a in range(original_len):
                        new_x[0,a] = x_[0,a]
                        
                    # print("new_x")
                    # print(new_x)
                        
                        
                    # print(x_.shape)
                    # print(x_.shape[0])
                    # print(x_.shape[1])
                    # print(x_.shape[2])
                    # print(x_.shape[3])
                    
                    input_seq = int(original_len/(15*3))
                    # print("input sequenz")
                    # print(input_seq)
                    
                    
                    new_x = np.reshape(new_x, (1, input_seq, 15, 3))
                    new_x = np.transpose(new_x, (0, 3, 1, 2))
                    new_x[:, 2, :, :] = 0
                    
                    a = np.zeros((1, 2, input_seq, 15))
                    
                    # a = np.zeros((128, 2, 32, 15))
                    # a = np.zeros((128, 2, 64, 15))
                    # a = np.zeros((128, 2, 32, 15))
                
                    a[:,0,:,:] = new_x[:,0,:,:]
                    a[:,1,:,:] = new_x[:,1,:,:]   
                    
                    
                    
                    
                    # burda normalize edebilir miyim :)
                    tmp = np.copy(a)
                    curr_min_x = np.min(a[:,0,:,:][a[:,0,:,:] > 0])
                    curr_min_y = np.min(a[:,1,:,:][a[:,1,:,:] > 0])

                    # Set x start to 0
                    a[:,0,:,:] = a[:,0,:,:] - curr_min_x
                    # Set y start to 0
                    a[:,1,:,:] = a[:,1,:,:] - curr_min_y
                    
                    
                    # Set x to max image_size.width
        
                    max_factor_x = 1 / (np.max(tmp[:,0, :, :])-curr_min_x)
                    max_factor_y = 1 / (np.max(tmp[:,1, :, :])-curr_min_y)                  
                    
        
                    # max_factor_x = 1 / np.max(ehpi_img[0, :, :])
                    # max_factor_y = 1 / np.max(ehpi_img[1, :, :])
                    a[:,0,:,:] = a[:,0,:,:] * max_factor_x
                    a[:,1,:,:] = a[:,1,:,:] * max_factor_y
                    a[:,0,:,:][tmp[:,0, :, :] == 0] = 0
                    a[:,1,:,:][tmp[:,1, :, :] == 0] = 0
                    # test = ehpi_img[0, :, :].max()
                    # test2 = ehpi_img[1, :, :].max()           

                    
                    
                    data_new_3 = np.transpose(a, (0,2,1,3))
                    
                    
                    # input_LSTM = data_new_3.reshape(128,32,30)
                    # input_LSTM = data_new_3.reshape(1,x_.shape[2],30)
                    # input_LSTM = data_new_3.reshape(128,64,30)
                    input_LSTM = data_new_3.reshape(1,input_seq,30)
                    
                    input_LSTM = torch.Tensor(input_LSTM).to(device)
                    
                    
                    print("index_val : " +str(i))
                    
                    # FOR LSTM 
                    h_2 = tuple([e.data for e in h_2])
                
                
                    # FOR GRU
                    # h_2 = h_2.data
                    
                    batch = 1
                    outputs,hidden = model(input_LSTM, h_2,batch,input_seq)
                    loss_test = loss_func(outputs, y)
                    losses_test.append(loss_test.item())
            
                loss_total_test = sum(losses_test) / len(losses_test)
                losses_out_test.append(loss_total_test)
            
                print("VALIDATION LOSS :")
                print("{}: {}".format(epoch, loss_total_test))          

            # if (epoch != 0 and epoch % train_config.checkpoint_epoch == 0):
            '''
            print("Let's do some testing")  
                
            if test_loader is not None:
                accuracy = self.test(model, test_loader=test_loader)
                accuracies_out.append(accuracy)
                print("TEST NORMAL :")
                print("{}: {}".format(epoch, accuracy))             
                # tb.add_scalar('Accuracy_Test_Sequenz',accuracy_seq,epoch)
                tb.add_scalar('Accuracy_Test_Normal',accuracy,epoch)
                tb.add_scalar('Accuracy_Test_VIDEO_VAR_LEN',accuracy,epoch)
                # tb.add_scalars('Accuracy Together',{'Accuracy_Clip' :accuracy, 'Accuracy_Video':accuracy_video},epoch)

            # SAVING IN DIFFERENT GRAPHS 
            # tb.add_scalar('Loss_train', loss_total,epoch)
            # tb.add_scalar('Loss_test', loss_total_test,epoch)
            
            # SAVING IN THE SAME GRAPH 
            # tb.add_scalars('LOSSES_blabla',{'Loss_train' :loss_total, 'Loss_test':loss_total_test},epoch)
            
            # if epoch % 15 == 0 :         
              #   torch.save(model.state_dict(), train_config.get_output_path(epoch = train_config.num_epochs, epoch_checkpoint = epoch))
        
        tb.close() # don't forget to close it 
        # torch.save(model.state_dict(), train_config.get_output_path(train_config.num_epochs))
                
        return losses_out, accuracies_out_seq ,losses_out_test,accuracies_out

    def test(self, model, test_loader: DataLoader):
        model.eval()
        corrects = []
        corrects_total = []
        corrects_total_video = []
        
        h_3 = model.init_hidden_train(1) 
        
        for i, data in enumerate(test_loader):
            x = Variable(data["x"]).to(device)
            y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
            lala = data["y"].numpy()
            seq = Variable(torch.tensor(data["seq"], dtype=torch.long)).to(device)
                    # print("seq")
                    # print(seq)
                    # print(seq[1])
                    
                    
                    
                    # print("input x to model : (in TEST)")
                    # print(x)
                    # print("output y")
                    # print(y)                   
            x_ = x.cpu().numpy()
            #print("X NEYMIS BAKALIM")
            #print(x_)
            #print(x_.shape)
                    
                    
                    
                    
            original_len = int(x_[0,13500])
                    # original_len = x_[-1]
            #print("original_len")
            #print(original_len)
                    
            new_x = np.zeros((1,original_len))
                    
                
            a = 0
            for a in range(original_len):
                new_x[0,a] = x_[0,a]
                        
           # print("new_x")
           # print(new_x)
                        
                        
                    # print(x_.shape)
                    # print(x_.shape[0])
                    # print(x_.shape[1])
                    # print(x_.shape[2])
                    # print(x_.shape[3])
                    
            input_seq = int(original_len/(15*3))
            #print("input sequenz")
            #print(input_seq)
                    
                    
            new_x = np.reshape(new_x, (1, input_seq, 15, 3))
            new_x = np.transpose(new_x, (0, 3, 1, 2))
            new_x[:, 2, :, :] = 0
                    
            a = np.zeros((1, 2, input_seq, 15))
                    
                    # a = np.zeros((128, 2, 32, 15))
                    # a = np.zeros((128, 2, 64, 15))
                    # a = np.zeros((128, 2, 32, 15))
                
            a[:,0,:,:] = new_x[:,0,:,:]
            a[:,1,:,:] = new_x[:,1,:,:]   
                    
                    
                    
                    
                    # burda normalize edebilir miyim :)
            tmp = np.copy(a)
            curr_min_x = np.min(a[:,0,:,:][a[:,0,:,:] > 0])
            curr_min_y = np.min(a[:,1,:,:][a[:,1,:,:] > 0])

                    # Set x start to 0
            a[:,0,:,:] = a[:,0,:,:] - curr_min_x
                    # Set y start to 0
            a[:,1,:,:] = a[:,1,:,:] - curr_min_y
                    
                    
                    # Set x to max image_size.width
        
            max_factor_x = 1 / (np.max(tmp[:,0, :, :])-curr_min_x)
            max_factor_y = 1 / (np.max(tmp[:,1, :, :])-curr_min_y)                  
                    
        
                    # max_factor_x = 1 / np.max(ehpi_img[0, :, :])
                    # max_factor_y = 1 / np.max(ehpi_img[1, :, :])
            a[:,0,:,:] = a[:,0,:,:] * max_factor_x
            a[:,1,:,:] = a[:,1,:,:] * max_factor_y
            a[:,0,:,:][tmp[:,0, :, :] == 0] = 0
            a[:,1,:,:][tmp[:,1, :, :] == 0] = 0
                    # test = ehpi_img[0, :, :].max()
                    # test2 = ehpi_img[1, :, :].max()           

                    
                    
            data_new_3 = np.transpose(a, (0,2,1,3))
                    
                    
                    # input_LSTM = data_new_3.reshape(128,32,30)
                    # input_LSTM = data_new_3.reshape(1,x_.shape[2],30)
                    # input_LSTM = data_new_3.reshape(128,64,30)
            input_LSTM = data_new_3.reshape(1,input_seq,30)
                    
            input_LSTM = torch.Tensor(input_LSTM).to(device)
                    
                    
            print("index_test : " +str(i))
            
            # FOR LSTM 
            h_3 = tuple([e.data for e in h_3])
                
            batch = 1
            outputs,hidden = model(input_LSTM, h_3,batch,input_seq)
            
            # outputs = outputs.data.cpu().numpy()[0]
            outputs = outputs.data.cpu().numpy()
            # print("outputs later")
            # print(outputs)
            # print(outputs.shape)
            # print(outputs[1:4])
            
            
            '''
            for i in range (127):        
                if i <= 121 : 
                    if i == 0 and (seq[i] == seq[i+6]):
                        prediction_sequence =  np.argmax(outputs[i:i+7], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+7])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/7
                        corrects_total_video.append(correct_rate)   
                    elif i == 0 and (seq[i] == seq[i+5]):
                        prediction_sequence =  np.argmax(outputs[i:i+6], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+6])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/6
                        corrects_total_video.append(correct_rate)             
                    elif i == 0 and (seq[i] == seq[i+4]):
                        prediction_sequence =  np.argmax(outputs[i:i+5], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+5])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/5
                        corrects_total_video.append(correct_rate)              
                    elif i == 0 and (seq[i] == seq[i+3]):
                        prediction_sequence =  np.argmax(outputs[i:i+4], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+4])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/4
                        corrects_total_video.append(correct_rate)             
                    elif i == 0 and (seq[i] == seq[i+2]):
                        prediction_sequence =  np.argmax(outputs[i:i+3], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+3])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/3
                        corrects_total_video.append(correct_rate)             
                    elif i == 0 and (seq[i] == seq[i+1]):
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)
                        
                         
                    elif i != 0 and (seq[i] == seq[i+6]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+7], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+7])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/7
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+5]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+6], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+6])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/6
                        corrects_total_video.append(correct_rate)
                    elif i != 0 and (seq[i] == seq[i+4]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+5], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+5])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/5
                        corrects_total_video.append(correct_rate)              
                    elif i != 0 and (seq[i] == seq[i+3]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+4], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+4])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/4
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+2]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+3], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+3])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/3
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+1]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)              

                elif i == 122:
                        
                    if i != 0 and (seq[i] == seq[i+5]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+6], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+6])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/6
                        corrects_total_video.append(correct_rate)
                    elif i != 0 and (seq[i] == seq[i+4]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+5], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+5])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/5
                        corrects_total_video.append(correct_rate)              
                    elif i != 0 and (seq[i] == seq[i+3]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+4], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+4])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/4
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+2]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+3], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+3])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/3
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+1]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)   
            
                elif i == 123:

                    if i != 0 and (seq[i] == seq[i+4]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+5], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+5])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/5
                        corrects_total_video.append(correct_rate)              
                    elif i != 0 and (seq[i] == seq[i+3]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+4], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+4])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/4
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+2]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+3], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+3])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/3
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+1]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)              
 
                elif i == 124:
             
                    if i != 0 and (seq[i] == seq[i+3]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+4], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+4])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/4
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+2]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+3], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+3])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/3
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+1]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)   
                        
                elif i == 125:
                        
                    if i != 0 and (seq[i] == seq[i+2]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+3], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+3])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/3
                        corrects_total_video.append(correct_rate)            
                    elif i != 0 and (seq[i] == seq[i+1]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)                       
                        
                elif i == 126:
                                 
                    if i != 0 and (seq[i] == seq[i+1]) and seq[i] != seq[i-1]:
                        prediction_sequence =  np.argmax(outputs[i:i+2], axis=1)
                        true_false = np.equal(prediction_sequence, lala[i:i+2])
                        true = np.sum(true_false)
                        correct_rate =  (true*100)/2
                        corrects_total_video.append(correct_rate)                           
                        
                        
            '''        
                        
                        
                        
            predictions = np.argmax(outputs, axis=1)
            # print(predictions)
            
            '''
            true_false = np.equal(predictions, lala)
            # print("true_false")
            # print(true_false)
            true = np.sum(true_false)
            # print("true")
            # print(true)
            correct_rate =  (true*100)/128
            # print("correct_rate")
            # print(correct_rate)
            
            corrects_total.append(correct_rate)
            '''
            
            # print("PRED")
            # print(predictions)
            # print("LALA")
            # print(lala)
            correct = predictions == lala
            corrects.append(int(correct))
            
            # print("correct akt")
            # print(correct)
            
        print("sum total")  
        print(float(sum(corrects)))   
        print(float(len(test_loader)))
        
        '''
        accuracy = float(sum(corrects_total)) / float(len(test_loader))
        accuracy_video = float(sum(corrects_total_video)) / float(len(corrects_total_video))
        print("sum(corrects_total_video)")
        print(float(sum(corrects_total_video)))
        print("len(corrects_total_video)")
        print(float(len(corrects_total_video)))
        
        print("accuracy video:")
        print(accuracy_video)
        '''
        
        accuracy = float(sum(corrects)) / float(len(test_loader))
        print("ACCURACY")
        print(accuracy)
        
        
        logger.error("Test set accuracy: {}".format(accuracy))
        return accuracy

    def test_by_seq(self, model, test_loader: DataLoader):
        model.eval()
        corrects = []
        sequence_labels: Dict[int, int] = {}
        sequence_results: Dict[int, List[int]] = {}
        label_count: Dict[int, int] = {}
        for i, data in enumerate(test_loader):
            x = Variable(data["x"]).to(device)
            y = data["y"].numpy()[0]
            seq = data["seq"].numpy()[0]
            outputs = model(x).data.cpu().numpy()[0]
            predictions = np.argmax(outputs)
            if seq not in sequence_results:
                sequence_results[seq] = []
                sequence_labels[seq] = y
                if y not in label_count:
                    label_count[y] = 0
                label_count[y] += 1
            sequence_results[seq].append(predictions)
        corrects_per_label: Dict[int, List[int]] = {}
        for sequence_id, predictions in sequence_results.items():
            prediction = max(set(predictions), key=predictions.count)
            label = sequence_labels[sequence_id]
            correct = prediction == label
            if label not in corrects_per_label:
                corrects_per_label[label] = []
            corrects_per_label[label].append(correct)
            corrects.append(int(correct))
        accuracy = float(sum(corrects)) / float(len(sequence_labels))
        logger.error("Test set accuracy: {} [Num: Test Sequences: {}]".format(accuracy, len(sequence_labels)))
        # for label, corrects in corrects_per_label.items():
        #     accuracy = sum(corrects) / label_count[label]
        #     logger.error("Label accuracy: {} [Label: {}, Num_Tests: {}]".format(accuracy, label, label_count[label]))
        return accuracy

        
        
'''
                    accuracy_seq = self.test_by_seq(model, test_loader=test_loader)
                    accuracies_out_seq.append(accuracy_seq)
                    print("TEST SEQUENZ :")
                    print("{}: {}".format(epoch, accuracy_seq))
'''