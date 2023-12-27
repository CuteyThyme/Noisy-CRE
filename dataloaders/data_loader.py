import torch
from torch.utils.data import Dataset, DataLoader

class data_set(Dataset):

    def __init__(self, data,config=None,is_train=True):
        self.data = data
        self.config = config
        self.bert = True
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):
        label = torch.tensor([item[0]['relation'] for item in data])
        tokens = [torch.tensor(item[0]['tokens']) for item in data]
        ind = torch.tensor([item[1] for item in data])
        if self.is_train is True:
            flag = torch.tensor([item[0]['relation'] == item[0]['ori_relation']for item in data])
            return (
                label,
                tokens,
                flag,
                ind
            )
        else:
            return (
                label,
                tokens,
                ind
            )
       

def get_data_loader(config, data, shuffle = False, drop_last = False, batch_size = None, is_train=True):

    dataset = data_set(data, config, is_train)

    if batch_size == None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader