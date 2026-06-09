from torch.utils.data import Dataset

class UserItemInterDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        user = self.data[index][0]
        pos_item = self.data[index][1]
        neg_item = self.data[index][2]
        return user, pos_item, neg_item
    def __len__(self):
        return len(self.data)
