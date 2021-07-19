from typing import Dict, List

import numpy as np
import torch
from nobos_commons.tools.log_handler import logger
from nobos_torch_lib.configs.training_configs.training_config_base import TrainingConfigBase
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainerEhpi(object):
    def train(self, train_loader: DataLoader, train_config: TrainingConfigBase, model, test_loader: DataLoader = None):
        print("Train model: {}".format(train_config.model_name))

        model.to('cuda')


        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=train_config.learning_rate, momentum=train_config.momentum, weight_decay=train_config.weight_decay)

        losses_out = []
        losses_out_test =[]
        
        accuracies_out = []
        accuracies_out_seq =[]

        for epoch in range(train_config.num_epochs):
		
            model.train()
            print("training model")
            train_config.learning_rate_scheduler(optimizer, epoch)
            
            losses = []
            losses_test = []
            
            for i, data in enumerate(train_loader):
                x = Variable(data["x"]).to(device)
                y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
                
                print("index_train : " +str(i))
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_func(outputs, y)
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

            loss_total = sum(losses) / len(losses)
            losses_out.append(loss_total)
            
            print("LOSS FOR TRAINING")
            print("{}: {}".format(epoch, loss_total))
            
            
            for i,data in enumerate(test_loader):
                x = Variable(data["x"]).to(device)
                y = Variable(torch.tensor(data["y"], dtype=torch.long)).to(device)
                
                print("index_test : " +str(i))
                optimizer.zero_grad()
                outputs = model(x)
                loss_test = loss_func(outputs, y)
                losses_test.append(loss_test.item())
            
            loss_total_test = sum(losses_test) / len(losses_test)
            losses_out_test.append(loss_total_test)
            
            print("LOSS FOR TEST")
            print("{}: {}".format(epoch, loss_total_test))          
            
			
            # train_config.checkpoint_epoch
            if epoch != 0 and epoch % 3 == 0:
                print("hi")
                if test_loader is not None:
                
                    accuracy_seq = self.test_by_seq(model, test_loader=test_loader)
                    accuracies_out_seq.append(accuracy_seq)
                    print("TEST SEQ :")
                    print("{}: {}".format(epoch, accuracy_seq))

                    accuracy = self.test(model, test_loader=test_loader)
                    accuracies_out.append(accuracy) 
                    print("TEST NORMAL :")
                    print("{}: {}".format(epoch, accuracy))                 

                    
                    
                    
        if test_loader is not None:
            self.test_by_seq(model, test_loader=test_loader)
            self.test(model, test_loader=test_loader)
            
        torch.save(model.state_dict(), train_config.get_output_path(train_config.num_epochs))
        return losses_out, accuracies_out_seq ,losses_out_test,accuracies_out

    def test(self, model, test_loader: DataLoader):
        model.eval()
        corrects = []
        for i, data in enumerate(test_loader):
            x = Variable(torch.tensor(data["x"], dtype=torch.float)).to(device)
            y = data["y"].numpy()[0]
            outputs = model(x).data.cpu().numpy()[0]
            predictions = np.argmax(outputs)
            correct = predictions == y
            corrects.append(int(correct))
        accuracy = float(sum(corrects)) / float(len(test_loader))
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
