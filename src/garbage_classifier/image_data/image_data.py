from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

class LazyDataset(Dataset):
    
    def __init__(self, dataset, transform=None):
        self.toTensor = transforms.ToTensor()
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        input = self.toTensor(self.dataset[index][0])
        x = self.transform(input) if self.transform else input
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)
    
class ImageData():
    
    def __init__(self, train_path, test_path, batch_size, transforms=None):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_data = ImageFolder(self.train_path)
        self.test_data = ImageFolder(self.test_path)
        self._train_loader = DataLoader(LazyDataset(self.train_data, transforms), batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._test_loader = DataLoader(LazyDataset(self.test_data, transforms), batch_size=self.batch_size, shuffle=False, num_workers=0)

    def get_train_loader(self):
        return self._train_loader

    def get_test_loader(self):
        return self._test_loader