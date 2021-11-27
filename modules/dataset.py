from torch.utils.data import Dataset

class FishnetDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.images = dataframe['data']
        self.labels = dataframe['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images.iloc[idx].reshape(3, 32, 32).transpose(1, 2, 0) # reshape [32, 32, 3]
        img = self.transform(img) # reshape [3, 32, 32]
        return img, self.labels.iloc[idx]
