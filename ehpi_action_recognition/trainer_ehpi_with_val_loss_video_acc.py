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
            
        for epoch in range(train_config.num_epochs):
        
            losses = []
            losses_test = []
            
            model.train()
            print("training model")
            
            train_config.learning_rate_scheduler(optimizer, epoch)
                      
            for i, data in enumerate(train_loader):
                x = Variable(data["x"]).to(device)
                y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
                
                print("index_train : " +str(i))
                optimizer.zero_grad()
                outputs = model(x)
                # print("outputs :")
                # print(outputs)
                loss = loss_func(outputs, y)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                
            # loss average overall the batches will be calculated
            loss_total = sum(losses) / len(losses)
            losses_out.append(loss_total)
            
            print("TRAINING LOSS :")
            print("{}: {}".format(epoch, loss_total))
            
            # for validation loss in each epoch : 
            with torch.no_grad():
                model.eval()
                print("validation model")   
                
                for i,data in enumerate(test_loader):
                    x = Variable(data["x"]).to(device)
                    y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
                
                    print("index_test : " +str(i))
                    outputs = model(x)
                    loss_test = loss_func(outputs, y)
                    losses_test.append(loss_test.item())
            
                loss_total_test = sum(losses_test) / len(losses_test)
                losses_out_test.append(loss_total_test)
            
                print("VALIDATION LOSS :")
                print("{}: {}".format(epoch, loss_total_test))          

            # if (epoch != 0 and epoch % train_config.checkpoint_epoch == 0):
            
            print("Let's do some testing")  
                
            if test_loader is not None:
                accuracy,accuracy_video = self.test(model, test_loader=test_loader)
                accuracies_out.append(accuracy) 
                print("TEST NORMAL :")
                print("{}: {}".format(epoch, accuracy))             
                # tb.add_scalar('Accuracy_Test_Sequenz',accuracy_seq,epoch)
                tb.add_scalar('Accuracy_Test_Normal',accuracy,epoch)
                tb.add_scalar('Accuracy_Test_VIDEO',accuracy_video,epoch)
                tb.add_scalars('Accuracy Together',{'Accuracy_Clip' :accuracy, 'Accuracy_Video':accuracy_video},epoch)      

            # SAVING IN DIFFERENT GRAPHS 
            # tb.add_scalar('Loss_train', loss_total,epoch)
            # tb.add_scalar('Loss_test', loss_total_test,epoch)
            
            # SAVING IN THE SAME GRAPH 
            tb.add_scalars('LOSSES_blabla',{'Loss_train' :loss_total, 'Loss_test':loss_total_test},epoch)
        
        torch.save(model.state_dict(), train_config.get_output_path(epoch = train_config.num_epochs, epoch_checkpoint = epoch))
        
        tb.close() # don't forget to close it 
        # torch.save(model.state_dict(), train_config.get_output_path(train_config.num_epochs))
                
        return losses_out, accuracies_out_seq ,losses_out_test,accuracies_out

    def test(self, model, test_loader: DataLoader):
        model.eval()
        corrects = []
        corrects_total = []
        corrects_total_video = []
        
        
        for i, data in enumerate(test_loader):
            x = Variable(torch.tensor(data["x"], dtype=torch.float)).to(device)
            lala = data["y"].numpy()
            seq = Variable(torch.tensor(data["seq"], dtype=torch.long)).to(device)
            seq = seq.cpu().numpy()
            #print(seq)
            #print(len(seq))
            
            
            # y = data["y"].numpy()[0]

            outputs = model(x).data.cpu().numpy()
            # outputs = outputs.data.cpu().numpy()
            
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
                                    
            predictions = np.argmax(outputs, axis=1)
            # print(predictions)
            
            
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

            
        # print("sum total")  
        # print(float(sum(corrects_total)))   
        accuracy = float(sum(corrects_total)) / (float(len(test_loader))*100)
        accuracy_video = float(sum(corrects_total_video)) / (float(len(corrects_total_video))*100)
        # print("sum(corrects_total_video)")
        # print(float(sum(corrects_total_video)))
        # print("len(corrects_total_video)")
        # print(float(len(corrects_total_video)))
        
        print("accuracy video:")
        print(accuracy_video)
        
        # accuracy = float(sum(corrects)) / float(len(test_loader))
        
        
        logger.error("Test set accuracy: {}".format(accuracy))
        return accuracy,accuracy_video

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