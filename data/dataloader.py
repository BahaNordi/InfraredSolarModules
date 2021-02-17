from torchvision import transforms
import torch
from torchvision.datasets import ImageFolder
import warnings

warnings.filterwarnings("ignore")


class SolarDataLoader(object):
    def __init__(self, config):
        self.train_dir = config['data']['train_dir']
        self.val_dir = config['data']['val_dir']
        self.batch_size = config['data']['batch_size']
        self.mean = config['data']['preprocessing']['mean']
        self.std = config['data']['preprocessing']['std']
        self.num_workers = config['data']['num_workers']
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    @property
    def train_loader(self):
        if not self._train_loader:
            train_transform = transforms.Compose([
                # transforms.Grayscale(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(180),
                # transforms.RandomCrop((40, 24), padding=(2, 2, 0, 0)),
                transforms.RandomApply(torch.nn.ModuleList([transforms.RandomCrop((40, 24), padding=(2, 2, 0, 0))]),
                                       p=0.5),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.05, contrast=0.05,
                                                                                   saturation=0.1, hue=0.05)]), p=0.3),
                transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur([3], sigma=(0.1, 2.0))]), p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([self.mean, self.mean, self.mean], [self.std,
                                                                         self.std,
                                                                         self.std])])  # transforms.Normalize(self.mean, self.std) #gray scale channel

            train = ImageFolder(self.train_dir, train_transform)

            weights = self.make_weights_for_balanced_classes(train.imgs, len(train.classes))
            weights = torch.DoubleTensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            self._train_loader = torch.utils.data.DataLoader(train, sampler=sampler, batch_size=self.batch_size,
                                                             num_workers=self.num_workers)
            # self._train_loader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True,
            #                                                  num_workers=self.num_workers)
        return self._train_loader

    @property
    def val_loader(self):
        if not self._val_loader:
            val_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([self.mean, self.mean, self.mean], [self.std,
                                                                                                         self.std,
                                                                                                         self.std])])  # transforms.Grayscale(),
            val_set = ImageFolder(self.val_dir, val_transform)
            self._val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=self.batch_size,
                                                           num_workers=self.num_workers)
        return self._val_loader

    def make_weights_for_balanced_classes(self, images, nclasses):
        count = [0] * nclasses
        for item in images:
            count[item[1]] += 1
        weight_per_class = [0.] * nclasses
        n = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = n / float(count[i])
        # manualy increased the weight of normal class from 2 to 10 to avoid under representing diversity
        # in the normal class
        weight_per_class[7] = 10.
        weight = [0] * len(images)
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight


class SolarTestDataLoader(object):
    def __init__(self, config, augmentation_index=0):
        self.val_dir = config['data']['val_dir']
        self.batch_size = config['data']['batch_size']
        self.mean = config['data']['preprocessing']['mean']
        self.std = config['data']['preprocessing']['std']
        self.num_workers = config['data']['num_workers']
        self._test_loader = None
        self.augmentation_index = augmentation_index
        self.transforms = {1: transforms.RandomHorizontalFlip(p=1),
                           2: transforms.RandomVerticalFlip(p=1),
                           3: transforms.RandomRotation(180),
                           # 4: transforms.CenterCrop(35),
                           4: transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.05),
                           5: transforms.GaussianBlur([3], sigma=(0.1, 2.0))}

        if augmentation_index == 0:
            self.transform = []
        else:
            self.transform = [self.transforms[augmentation_index]]

    @property
    def test_loader(self):
        if not self._test_loader:
            test_transform = transforms.Compose(self.transform + [transforms.ToTensor(),
                                                transforms.Normalize([self.mean, self.mean, self.mean],
                                                [self.std, self.std, self.std])])

            test_set = ImageFolder(self.val_dir, test_transform)
            self._test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=self.batch_size,
                                                            num_workers=self.num_workers)
        return self._test_loader
