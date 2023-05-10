from torchvision import datasets, transforms
from base import BaseDataLoader

class ViTDataLoader(BaseDataLoader):
    """
    ViT data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256)
        ])
        self.data_dir = data_dir
        self.dataset = datasets.STL10(self.data_dir, split='train', download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)