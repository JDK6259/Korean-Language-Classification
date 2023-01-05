from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def split_data(dataset,input_ids, attention_masks, labels):
    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def ret_dataloader(train_dataset, val_dataset, batch_size):
    print('batch_size = ', batch_size)
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size 
            )
    return train_dataloader,validation_dataloader