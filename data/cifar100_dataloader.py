import torchvision
from torchvision import transforms
import torch
import warnings

warnings.filterwarnings("ignore")


class CIFARDataLoader(object):
    def __init__(self, config):
        self.cifar100_train_dir = config['data']['cifar100_train_dir']
        self.cifar100_val_dir = config['data']['cifar100_val_dir']
        self.batch_size = config['data']['batch_size']
        self.mean_rgb = [0.507, 0.487, 0.441]
        self.std_rgb = [0.267, 0.256, 0.276]
        self.num_workers = config['data']['num_workers']
        self._train_loader = None
        self._val_loader = None

    @property
    def train_loader(self):
        if not self._train_loader:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),  # randomly flip and rotate
                transforms.ToTensor(),
                transforms.Normalize(self.mean_rgb, self.std_rgb)])
            train = torchvision.datasets.CIFAR100(self.cifar100_train_dir, train=True, download=True,
                                                  transform=train_transform)
            self._train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True,
                                                             num_workers=self.num_workers)
        return self._train_loader

    @property
    def val_loader(self):
        if not self._val_loader:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean_rgb, self.std_rgb)])
            val_set = torchvision.datasets.CIFAR100(self.cifar100_val_dir, train=False, download=True,
                                                    transform=val_transform)
            self._val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=self.batch_size,
                                                           num_workers=self.num_workers)
        return self._val_loader
