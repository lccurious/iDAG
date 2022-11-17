# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms as T
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "NICOAnimal",
    "NICOVehicle",
    "NICOMixed",
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 4  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.datasets[index]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,)),
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ["0", "1", "2"]


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ["0", "1", "2"]


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape, num_classes):
        """
        Args:
            root: root dir for saving MNIST dataset
            environments: env properties for each dataset
            dataset_transform: dataset generator function
        """
        super().__init__()
        if root is None:
            raise ValueError("Data directory not specified!")

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.environments = environments

        for i in range(len(environments)):
            images = original_images[i :: len(environments)]
            labels = original_labels[i :: len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["-90%", "+90%", "+80%"]

    def __init__(self, root):
        super(ColoredMNIST, self).__init__(
            root,
            [0.9, 0.1, 0.2],
            self.color_dataset,
            (2, 14, 14),
            2,
        )

    def color_dataset(self, images, labels, environment):
        # Subsample 2x for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels, self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels, self.torch_bernoulli_(environment, len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ["0", "15", "30", "45", "60", "75"]

    def __init__(self, root):
        super(RotatedMNIST, self).__init__(
            root,
            [0, 15, 30, 45, 60, 75],
            self.rotate_dataset,
            (1, 28, 28),
            10,
        )

    def rotate_dataset(self, images, labels, angle):
        rotation = T.Compose(
            [
                T.ToPILImage(),
                T.Lambda(lambda x: rotate(x, angle, fill=(0,), resample=Image.BICUBIC)),
                T.ToTensor(),
            ]
        )

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        self.datasets = []
        for environment in environments:
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    N_STEPS = 15001
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]

    def __init__(self, root):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir)


class NICOMixedEnvironment(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_file_path, input_shape, transform):
        super().__init__()
        self.label_dict = {'animal': 0, 'vehicle': 1}
        self.transform = transform
        self.img_paths = []
        self.targets = []
        with open(csv_file_path) as f:
            for line in f.readlines():
                img_path, category_name, context_name, superclass = line.strip().split(',')
                img_path = img_path.replace('\\', '/')
                self.img_paths.append(f'{images_root}/{superclass}/images/{img_path}')
                self.targets.append(self.label_dict[superclass])
                
    def __len__(self):
        return len(self.targets)
                
    def __getitem__(self, key):
        with open(self.img_paths[key], 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.transform(image)
        return image, self.targets[key]
    
    
class NICOMixed(MultipleDomainDataset):
    # ENVIRONMENTS = ["train1", "train2", "train3", "train4", "val", "test"]
    ENVIRONMENTS = ["train1", "train2", "val", "test"]
    CHECKPOINT_FREQ = 200
    def __init__(self, root, test_envs, hparams):
        self.input_shape = (3, 224, 224)
        self.datasets = []
        self.num_classes = 2
        
        normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform = T.Compose([
            T.Resize((int(self.input_shape[1] / 0.875), int(self.input_shape[2] / 0.875))),
            T.CenterCrop(self.input_shape[1]),
            T.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            self.input_shape[1], normalize, hparams.get('data_augmentation_scheme', 'domainbed'))
        
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            csv_file_path = os.path.join(f'{root}/NICO/mixed_split_corrected/env_{env_name}.csv')
            self.datasets.append(NICOMixedEnvironment(f'{root}/NICO', csv_file_path, self.input_shape, env_transform))
        
        
class NICOEnvironment(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_file_path, input_shape, transform, category_dict):
        super().__init__()
        self.transform = transform
        
        self.img_paths = []
        self.targets = []
        with open(csv_file_path) as f:
            for line in f.readlines():
                img_path, category_name, context_name = line.strip().split(',')[:3]
                img_path = img_path.replace('\\', '/')
                self.img_paths.append(f'{images_root}/{img_path}')
                self.targets.append(category_dict[category_name])
                
    def __len__(self):
        return len(self.targets)
                
    def __getitem__(self, key):
        with open(self.img_paths[key], 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.transform(image)
        return image, self.targets[key]
    
        
class NICOMultipleDomainDataset(MultipleDomainDataset):
    # ENVIRONMENTS = [f"train{i}" for i in range(20)] + ["val", "test"]
    # N_WORKERS = 2
    ENVIRONMENTS = ["domain_1", "domain_2", "domain_3", "domain_4", "domain_val"]
    CHECKPOINT_FREQ = 1000
    def __init__(self, root, superclass, test_envs, category_dict, hparams):
        self.input_shape = (3, 224, 224)
        self.datasets = []
        
        if superclass == 'animal':
            normalize = T.Normalize(
                    mean=[0.408, 0.421, 0.412], std=[0.186, 0.191, 0.209])
        else:
            normalize = T.Normalize(
                    mean=[0.624, 0.609, 0.607], std=[0.220, 0.219, 0.211])
        
        transform = T.Compose([
            T.Resize((int(self.input_shape[1] / 0.875), int(self.input_shape[2] / 0.875))),
            T.CenterCrop(self.input_shape[1]),
            T.ToTensor(),
            normalize
        ])

        augment_transform = self.get_transform(
            self.input_shape[1], normalize, hparams.get('data_augmentation_scheme', 'domainbed'))
            
        split_name = hparams['nico_split_name']
        
        for i, env_name in enumerate(self.ENVIRONMENTS):
            if hparams['data_augmentation'] and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform
            csv_file_path = os.path.join(f'/home/ma-user/work/IIRM/NICO_settings/{split_name}/{superclass}_{env_name}.csv')
            self.datasets.append(NICOEnvironment(f'{root}/nico/{superclass}/images', csv_file_path, self.input_shape, env_transform, category_dict))
            
            
class NICOAnimal(NICOMultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        image_folders = ['bear', 'bird', 'cat', 'cow', 'dog', 'elephant', 'horse', 'monkey', 'rat', 'sheep']
        self.num_classes = len(image_folders)
        # NOTE: use the following for the original classes:
        # category_dict = {name: i for i, name in enumerate(image_folders)}
        # NOTE: use the following for 2 classes:
        category_dict = {'0': 0, '1': 1}
        super().__init__(root, 'animal', test_envs, category_dict, hparams)
            

class NICOVehicle(NICOMultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        image_folders = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'helicopter', 'motorcycle', 'train', 'truck']
        self.num_classes = len(image_folders)
        category_dict = {name: i for i, name in enumerate(image_folders)}
        super().__init__(root, 'vehicle', test_envs, category_dict, hparams)
