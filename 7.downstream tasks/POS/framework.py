from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.utils.data import DataLoader
import os
from seqeval.metrics import f1_score, classification_report
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2020)

class Framework:

    def __init__(self, args):
        self.args = args
 
    def train_step(self, batch_data, model):
        model.train()
        batch = tuple(t.to(self.args.device) for t in batch_data)
        input_ids, attention_mask, pred_mask, labels = batch
        predicted, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pred_mask=pred_mask,
            input_labels=labels
        )
        loss.backward()


        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()  # Update learning rate schedule

        return loss.item()
        

    def train(self, train_dataset, model, labels):
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size, shuffle=True)

        # get optimizer schedule and loss
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {   "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(train_dataloader) * self.args.num_train_epochs)
        
        # Train!
        print("***** Running training *****")
        print("  Num examples = ", len(train_dataset))
        print("  Num Epochs = ", self.args.num_train_epochs)

        # Check if continuing training from a checkpoint
        for epoch in range(0, int(self.args.num_train_epochs)):
            for step, batch in enumerate(train_dataloader):
                loss = self.train_step(batch, model)
            print('Train Epoch{} - loss: {:.6f}  '.format(epoch+1, loss))  # , accuracy, corrects

            self.args.save_model = self.args.save_model.split('pt')[0]+'pt'+str(epoch+1)
            print("Saving model checkpoint to %s"%self.args.save_model)
            torch.save(model.state_dict(), self.args.save_model)



    def test(self, test_dataset, model, all_labels):
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size, shuffle=False)
        model.to(self.args.device)
        model.eval()
        predicted_list = []
        labels_list = []
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                input_ids, attention_mask, pred_mask, labels = batch
                predicted = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pred_mask=pred_mask
                )
                predicted = predicted[0]

                if self.args.crf:
                    predicted = [seq[seq>=0].tolist() for seq in predicted]
                else:
                    predicted = [seq[mask==1].tolist() for seq, mask in zip(predicted, pred_mask)]

                groud_labels = [seq[mask==1].tolist() for seq, mask in zip(labels, pred_mask)]

                for tl, pl in zip(groud_labels, predicted):
                    labels_list.append([all_labels[l] for l in tl])
                    predicted_list.append([all_labels[l] for l in pl])
                
        with open(self.args.output_dir, "w", encoding="utf-8") as f:
            for labels in predicted_list:
                for l in labels:
                    f.write(l+"\n")
                f.write("\n")
        '''
        class_report = classification_report(labels_list, predicted_list, digits=4)
        print(class_report)      
        with open(self.args.output_dir+"_report", "w", encoding="utf-8") as f:
            f.write(class_report)
        '''
        #metrics
        n1, n2, n3 = [0,0,0]
        for i in range(len(predicted_list)):
            for j in range(len(predicted_list[i])):
                if predicted_list[i][j][0] == 'B':
                    n1 += 1
                if labels_list[i][j][0] == 'B':
                    n2 += 1
                if predicted_list[i][j][0] == 'B' and predicted_list[i][j] ==labels_list[i][j]:
                    if j == len(predicted_list[i])-1:
                        n3 += 1
                    else:
                        for k in range(j+1,len(predicted_list[i])):
                            if (predicted_list[i][k][0] == 'B'and labels_list[i][k][0] != 'B') or (predicted_list[i][k][0] != 'B'and labels_list[i][k][0] == 'B'):
                                break
                            if predicted_list[i][k] != labels_list[i][k]:
                                break
                            if predicted_list[i][k][0] == 'B' and labels_list[i][k][0] == 'B' and predicted_list[i][k] == labels_list[i][k]:
                                n3 +=1
                                break
                        else:
                            n3 += 1

        precision = n3/n1*100
        recall = n3/n2*100
        f1 = 2*precision*recall/(precision+recall)
        print("precision:", precision)
        print("recall:", recall)
        print("f1:",f1)
        with open(self.args.output_dir+"_report", "w", encoding="utf-8") as f:
            f.write("precision:"+str(precision)+'\n')
            f.write("recall:"+str(recall)+'\n')
            f.write("f1:"+str(f1)+'\n')

        with open('result.txt','a',encoding='utf-8') as fp:
            fp.write('\t'+str(f1)+'\n')
