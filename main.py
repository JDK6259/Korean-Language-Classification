import wandb
import time
import torch
import numpy as np
import pandas as pd

from util import *
from dataloader import *
from optimizer import *
from scheduler import *
from model import *

save_path = "model_path/model.pt"

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'lr': {
            'values' : [1e-5,2e-5,1e-3]
        },
        'batch_size': {
            'values': [32, 64]
        },
        'epochs':{
            'values':[4,5,6,7,8]
        },
        'optimizer': {
            'values': ['AdamW','Adam','AdamP']
        },
        'dp' : {
            'values'  : [0.1,0.2,0.3]
        },
        "scheduler": {
            'values' : ['linear','cosine']
        },
        'warmup_steps' : {
            'values' : [0, 100]
        }
    }

sweep_config['parameters'] = parameters_dict

import pprint

pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="kobert")

def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        seed_everything(42)
        dataset = pd.read_csv('/content/drive/MyDrive/Korean_Language_Classification/data/train.csv')
        dataset = dataset[['문장','유형']]
        dataset.columns = ['sentence', 'label']
        encoding = {"사실형": 0, "추론형": 1, "대화형": 2, "예측형": 3}
        dataset['label'] = dataset['label'].map(encoding)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = ret_model(dataset)
        print("get model")
        model.to(device)

        tokenizer = get_tokenizer()
        print("download tokenizer..")    

        input_ids, attention_masks, labels = sentence_to_token(dataset, tokenizer)
        print("finish tokenize")

        train_dataset, val_dataset = split_data(dataset, input_ids, attention_masks, labels)
        print("finish data split")

        train_dataloader, validation_dataloader = ret_dataloader(train_dataset, val_dataset, config.batch_size)
        total_steps = len(train_dataloader)
        print("get dataloader")

        optimizer = ret_optim(model, config.lr, config.optimizer)
        print("get optimizer")

        scheduler, epochs = ret_scheduler(optimizer, config.epochs, config.scheduler, config.warmup_steps, config.lr, total_steps)
        print("get shceduler")
        
        training_stats = []

        total_t0 = time.time()
        
        for epoch_i in range(0,epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0

            model.train()

            for step, batch in enumerate(train_dataloader):

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()        

                outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
                loss, logits = outputs['loss'], outputs['logits']
                wandb.log({'train_batch_loss':loss.item()})
                total_train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)            

            training_time = format_time(time.time() - t0)

            wandb.log({'avg_train_loss':avg_train_loss})

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()

            model.eval()


            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            for batch in validation_dataloader:
                
                b_input_ids = batch[0].cuda()
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                
                with torch.no_grad():        
                    outputs = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                    loss, logits = outputs['loss'], outputs['logits']
                    
                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)
                

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            validation_time = format_time(time.time() - t0)
            wandb.log({'val_accuracy':avg_val_accuracy,'avg_val_loss':avg_val_loss})
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        torch.save(model.state_dict(), +save_path)

wandb.agent(sweep_id, function=main, count=20)
print("Done")