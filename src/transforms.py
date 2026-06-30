import torch
from torchvision import transforms

from PIL import ImageFilter, ImageOps
import random
from torchvision.transforms import v2
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

def get_dataset_size(dataset: str):
    """Get corresponding image size for each dataset."""

    if dataset in ['cifar100', 'cifar100-irregular', 'cifar10', 'svhn']:
        return 32
    elif dataset == 'tinyimagenet':
        return 64
    elif dataset in ['imagenet100','imagenet200', 'imagenet', 'clear100','imagenet100-irregular']:
        return 224
    elif dataset in ['imagenet32', 'imagenet32-100', 'imagenet32-200']:
        return 32
    elif dataset == 'cars':
        return 224
    else:
        raise ValueError("Dataset not supported.")


def get_dataset_normalize(dataset: str):
    if dataset in ['cifar100','cifar100-irregular'] :
        return v2.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        )
    elif dataset == 'cifar10':
        return v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif dataset == 'tinyimagenet':
        return v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    elif dataset in ['imagenet100','imagenet200', 'imagenet', 'imagenet30', 'imagenet100-irregular']:
        return v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    elif dataset in ['imagenet32', 'imagenet32-100', 'imagenet32-200']:
        return v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    elif dataset in ['clear10', 'clear100']:
        return v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    elif dataset == 'svhn':
        return v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    elif dataset == 'cars':
        return v2.Normalize(
            (0.4707, 0.4602, 0.4550), (0.2638, 0.2629, 0.2678)
    )

    else:
        raise ValueError(f'Normalization for dataset "{dataset}" not supported')

def get_dataset_denormalize(dataset: str):
    if dataset in ['cifar100','cifar100-irregular'] :
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)

    elif dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    elif dataset == 'tinyimagenet':
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    elif dataset in ['imagenet100','imagenet200', 'imagenet', 'imagenet30', 'imagenet100-irregular']:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    elif dataset in ['imagenet32', 'imagenet32-100', 'imagenet32-200']:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    elif dataset in ['clear10', 'clear100']:
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    elif dataset == 'svhn':
        mean = (0.5, 0.5, 0.5)
        std  = (0.5, 0.5, 0.5)

    elif dataset == 'cars':
        mean = (0.4707, 0.4602, 0.4550)
        std  = (0.2638, 0.2629, 0.2678)

    else:
        raise ValueError(f'Denormalization for dataset "{dataset}" not supported')

    return v2.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std]
    )


def get_dataset_transforms(dataset: str):
    """Get corresponding normalization transform for each dataset."""

    if dataset in ['cifar100','cifar100-irregular'] :
        return v2.Compose([
            v2.ToTensor(),
            notNaN(),
            get_dataset_normalize(dataset)])
    elif dataset == 'cifar10':
        return v2.Compose([
            v2.ToTensor(),
            notNaN(),
            get_dataset_normalize(dataset)])
    elif dataset == 'tinyimagenet':
        return v2.Compose([
            v2.ToTensor(),
            notNaN(),
            get_dataset_normalize(dataset)])
    elif dataset in ['imagenet100','imagenet200', 'imagenet', 'imagenet30', 'imagenet100-irregular']:
        return transforms.Compose([
            v2.ToTensor(),
            notNaN(),
            v2.Resize((224,224)),
            get_dataset_normalize(dataset)])
    elif dataset in ['imagenet32', 'imagenet32-100', 'imagenet32-200']:
        return v2.Compose([
            v2.Resize((32, 32)),
            v2.ToTensor(),
            notNaN(),
            get_dataset_normalize(dataset)])
    elif dataset in ['clear10', 'clear100']:
        return v2.Compose([
            v2.ToTensor(),notNaN(),
            v2.Resize(224),
            v2.CenterCrop(224),
            get_dataset_normalize(dataset)])
    elif dataset == 'svhn':
        return v2.Compose([
                    v2.Resize(32),
                    v2.ToTensor(),notNaN(),
                    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == 'cars':
        return v2.Compose([
            v2.Resize((224, 224)),
            v2.ToTensor(),notNaN(),
            get_dataset_normalize(dataset)
    ]) 
    

    else:
        raise ValueError(f'Base Transforms for dataset "{dataset}" not supported')


def get_dataset_crop(dataset: str):
    """Get corresponding crop transform for each dataset."""

    if dataset in ['cifar100','cifar100-irregular'] :
        # return transforms.RandomCrop(32, padding=4),
        return v2.RandomResizedCrop(32, scale=(0.2, 1.))
    elif dataset == 'cifar10':
        # return transforms.RandomCrop(32, padding=4)
        return v2.RandomResizedCrop(32, scale=(0.2, 1.))
    elif dataset in ['imagenet100','imagenet200', 'imagenet', 'clear100']:
        # return transforms.RandomCrop(64, padding=8)
        return v2.RandomResizedCrop(224, scale=(0.2, 1.))
    elif dataset in ['imagenet32', 'imagenet32-100', 'imagenet32-200']:
        return v2.RandomResizedCrop(32, scale=(0.2, 1.))
    else:
        raise ValueError("Dataset not supported.")

 
class MultipleCropsTransform:
    """Take N random augmented views of one image."""
    def __init__(self, base_transform, n_crops=20):
        self.base_transform = base_transform
        self.n_crops = n_crops

    def __call__(self, x):
        stacked_views = [] # List of tensors, each contains one view for all samples in x
        for _ in range(self.n_crops):
            view_list = self.base_transform(x)
            stacked_views.append(view_list)
        return stacked_views



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        
def clamp_transform(image):
    # Clamping operation here
    return torch.clamp(image, min=0, max=1)

def get_transforms_simsiam(dataset: str = 'cifar100', crop=True):
    """Returns SimSiam augmentations with dataset specific crop."""

    all_transforms = [
        v2.RandomApply(
            [v2.Lambda(clamp_transform),
                v2.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4, hue=0.1)]
                                        , p=0.8),
        v2.RandomGrayscale(p=0.2),
        # transforms.RandomApply([transforms.GaussianBlur([.1, 2.])], p=0.5),
        v2.RandomHorizontalFlip()
    ]

    if crop:
        all_transforms = [get_dataset_crop(dataset)]+all_transforms
    return all_transforms


def get_transforms_barlow_twins(dataset: str = 'cifar100', crop=True):
    """Returns Barlow Twins augmentations with dataset specific crop."""
    all_transforms = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [v2.Lambda(clamp_transform),
                v2.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            v2.RandomGrayscale(p=0.2),
            #GaussianBlur(p=1.0),
            #Solarization(p=0.0),
        ]
    if crop:
        all_transforms = [get_dataset_crop(dataset)]+all_transforms
    return all_transforms

def get_transforms_simclr(dataset: str = 'cifar100', crop=True):
    img_size = get_dataset_size(dataset)
    # SimCLR paper: use (0.2, 1.0) for small images; (0.08, 1.0) is the ImageNet default
    crop_scale = (0.2, 1.0) if img_size <= 64 else (0.08, 1.0)
    # SimCLR paper: skip GaussianBlur for small images
    use_blur = img_size > 64

    transform = [get_dataset_denormalize(dataset)]
    if crop:
        transform.append(v2.RandomResizedCrop(size=img_size, scale=crop_scale, antialias=True))
    transform += [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.6, 0.6, 0.6, 0.2)], p=0.8),
        v2.RandomGrayscale(p=0.2),
    ]
    if use_blur:
        transform.append(v2.RandomApply([v2.GaussianBlur(5, (0.1, 2.0))], 0.2))
    transform += [notNaN(), get_dataset_normalize(dataset)]
    return transform

def get_transforms_byol(dataset: str = 'cifar100'):
    """Returns BYOL augmentations with dataset specific crop."""
    all_transforms = [
        get_dataset_denormalize(dataset),
            
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.Lambda(clamp_transform),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1),
                                        ],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(sigma=[.1, 2.]),
        get_dataset_normalize(dataset),
        ]
    return all_transforms

def get_transforms_emp(dataset: str = 'cifar100'):
    """Returns EMP augmentations with dataset specific crop."""
    normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

    if dataset in ['cifar10', 'cifar100']:
        blur_kernel = 5
        crop = transforms.RandomResizedCrop(32,scale=(0.25, 0.25), ratio=(1,1))
        
    elif dataset in ['imagenet', 'imagenet100']:
        blur_kernel = 23 # Same as SwAV
        transforms.RandomResizedCrop(224, scale=(0.25, 0.25),
                                     interpolation=transforms.InterpolationMode.BICUBIC)
        
    
    all_transforms = [
        get_dataset_denormalize(dataset),
        crop,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel)], p=0.1),
        transforms.RandomSolarize(threshold=0.5 ,p=0.1), # threshold chosen from PIL solarize implementation
        normalize
    ]
    return all_transforms

    

class STL_transform:
    """
    Custom STL10 dataset for handling aligned and invariant transformations.
    """
    def __init__(
        self,
        dataset
    ):
        self.base_transform = transforms.ToTensor()
        self.aligned_transform = torch.nn.Sequential(
        v2.RandomResizedCrop(size=224, scale=(0.08, 1.0), antialias=True),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
            p=0.8
        ),
        v2.RandomGrayscale(p=0.2),
        notNaN()
    )
        self.invariant_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.5
        ),
        #get_dataset_transforms(dataset)
    ])

    def __call__(self, batch):
        half_batch_size = len(batch) // 2
        device = batch[0].device
        batched_item_x1 = []
        batched_item_x2 = []

        for i in range(half_batch_size):
            
            img1 = batch[2 * i].cpu().numpy()
            img2 = batch[2 * i + 1].cpu().numpy()

            

            # Base transform
            if self.base_transform:
                img1 = np.transpose(img1, (1, 2, 0))
                img2 = np.transpose(img2, (1, 2, 0))
                img11 = self.base_transform(img1)
                img12 = self.base_transform(img1)
                img21 = self.base_transform(img2)
                img22 = self.base_transform(img2)
            else:
                img11, img12 = torch.Tensor(img1), torch.Tensor(img1)
                img21, img22 = torch.Tensor(img2), torch.Tensor(img2)

            # Aligned transform
            if self.aligned_transform:
                paired_img1 = torch.stack([img11, img21])
                paired_img2 = torch.stack([img12, img22])

                paired_img1 = self.aligned_transform(paired_img1)
                paired_img2 = self.aligned_transform(paired_img2)

                img11, img12 = paired_img1[0], paired_img2[0]
                img21, img22 = paired_img1[1], paired_img2[1]

            # Invariant transform
            if self.invariant_transform:
                img11 = self.invariant_transform(img11)
                img12 = self.invariant_transform(img12)
                img21 = self.invariant_transform(img21)
                img22 = self.invariant_transform(img22)

            batched_item_x1 += [img11.to(device), img21.to(device)]
            batched_item_x2 += [img12.to(device), img22.to(device)]

        # ---- ICI : on renvoie des tensors au lieu de listes ----
        batched_item_x1 = torch.stack(batched_item_x1, dim=0)
        batched_item_x2 = torch.stack(batched_item_x2, dim=0)
        

        return (batched_item_x1, batched_item_x2)



class notNaN(nn.Module):
    def __init__(self):
        super(notNaN, self).__init__()
    def forward(self, x):
        x=torch.nan_to_num(x, nan=0.0)
        return x


def get_common_transforms(dataset: str = 'cifar100'):
    "Common transforms for self supervised models for better comparison"
    all_transforms = [
        get_dataset_denormalize(dataset),
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(clamp_transform),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        get_dataset_normalize(dataset),
        ]

    return all_transforms
    


def get_transforms(dataset: str, model: str, n_crops: int = 2):
    """Returns augmentations for self supervised models"""

    if model == "simsiam":
        all_transforms = get_transforms_simsiam(dataset)

    elif model == "barlow_twins":
        all_transforms = get_transforms_barlow_twins(dataset)

    elif model == "byol":
        all_transforms = get_transforms_byol(dataset)

    elif model in ['emp', 'simsiam_multiview', 'byol_multiview']:
        all_transforms = get_transforms_emp(dataset)

    elif model == "common":
        all_transforms = get_common_transforms(dataset)

    elif model == "stl":
        return STL_transform(dataset)

    elif model in ["simclr", "SCALE"]:
        all_transforms = get_transforms_simclr(dataset)

    else:
        raise ValueError(f"Model {model} not supported")

    return MultipleCropsTransform(v2.Compose(all_transforms), n_crops=n_crops)
