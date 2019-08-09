import torch

from PIL import Image
import torchvision.transforms.functional as tt
import os


class dataset_for_denoising(torch.utils.data.Dataset):
    """
    Load images and add noise
    """
    def __init__(self, pathdir, split='training', img_size=[250, 250], grayscale=False, clip_to_realistic=True,
                 augmentations=False, normalize=False, noise_std=0.25, resize=True):

        # Input:
        self.path = pathdir

        self.split = split
        self.img_size = img_size
        self.augmentations = augmentations
        self.resize = resize
        self.normalize = normalize

        # Dataset details:
        self.mean_rgb = [0.0, 0.0, 0.0]
        self.std_rgb = [1.0, 1.0, 1.0]
        self.noise_std = noise_std
        self.grayscale = grayscale
        self.clip_to_realistic = clip_to_realistic

    def __getitem__(self, index):
        # PIL image loading
        if self.resize:
            img = Image.open(self.image_path(index)).resize(self.img_size[::-1], Image.LANCZOS)
        else:
            img = Image.open(self.image_path(index))

        if self.grayscale:
            img = img.convert('L')

        # PIL augmentations
        if self.augmentations:
            img = self.augmentations(img)

        # Move to torch
        img = tt.to_tensor(img)

        # Add noise
        img_noise = img + self.noise_std * torch.randn_like(img)

        # Clip to [0,1]
        if self.clip_to_realistic:
            img.clamp_(0, 1)
            img_noise.clamp_(0, 1)

        if self.normalize:
            tt.normalize(img, self.mean_rgb, self.std_rgb)
            tt.normalize(img_noise, self.mean_rgb, self.std_rgb)

        return img, img_noise

    def image_path(self, index):
        raise NotImplementedError()

    def augment(self, img):
        """
        todo: augmentations
        """
        return img

    def unnormalize(self, img):
        """
        Remove normalization for visual purposes
        """
        if self.normalize:
            for t, m, s in zip(img, self.mean_rgb, self.std_rgb):
                t.mul_(s).add_(m).clamp_(0, 1)
        return img


class VOC_for_denoising(dataset_for_denoising):
    """
    Load VOC images and add noise
    """
    def __init__(self, pathdir, split='training', img_size=[250, 250], grayscale=False, clip_to_realistic=True,
                 augmentations=False, normalize=True, noise_std=0.25, resize=True):
        super().__init__(pathdir, split, img_size, grayscale, clip_to_realistic, augmentations,
                         normalize, noise_std, resize)

        # Dataset parameters
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

        # Get files and dataset folders
        if split == 'training':
            dict_file = os.path.join(self.path, 'ImageSets', 'Segmentation', 'train.txt')
            with open(dict_file, 'r') as file:
                self.files = file.read().splitlines()
        else:
            dict_file = os.path.join(self.path, 'ImageSets', 'Segmentation', 'val.txt')
            with open(dict_file, 'r') as file:
                self.files = file.read().splitlines()

    def __len__(self):
        """
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        """
        return len(self.files)

    def image_path(self, index):
        return os.path.join(self.path, 'JPEGImages', self.files[index] + '.jpg')


class BSDS300_for_denoising(dataset_for_denoising):
    """
    Load BSDS images and add noise
    """
    def __init__(self, pathdir, split='training', img_size=[321, 481], grayscale=False, clip_to_realistic=True,
                 augmentations=False, normalize=False, noise_std=0.25, resize=False, flip=False):
        super().__init__(pathdir, split, img_size, grayscale, clip_to_realistic,
                         augmentations, normalize, noise_std, resize)

        # Dataset parameters
        self.mean_rgb = [0.43423736, 0.44302556, 0.36703733]
        self.std_rgb = [0.24956392, 0.2343471 , 0.2418366]
        self.flip = flip

        # Get files and dataset folders
        if split == 'training':
            dict_file = os.path.join(self.path, 'iids_train.txt')
            with open(dict_file, 'r') as file:
                self.files = file.read().splitlines()
            self.folder = 'train'
            self.format = '.jpg'
        elif split == 'trainingZhang':
            dict_file = os.path.join(self.path, 'iids_train.txt')
            with open(dict_file, 'r') as file:
                self.files = file.read().splitlines()
            self.folder = 'train_gray_matlab'
            self.format = '.png'
        elif split == 'testing':
            dict_file = os.path.join(self.path, 'iids_test.txt')
            with open(dict_file, 'r') as file:
                self.files = file.read().splitlines()
            self.folder = 'test'
            self.format = '.jpg'
        elif split == 'testing68':
            self.folder = 'test68'
            self.files = os.listdir(os.path.join(self.path, 'images', self.folder))
            self.format = ''
        elif split == 'testing68Zhang':
            self.folder = 'testzhang'
            self.files = os.listdir(os.path.join(self.path, 'images', self.folder))
            self.format = ''
        elif split == 'testing68Roth':
            dict_file = os.path.join(self.path, 'foe_test.txt')
            with open(dict_file, 'r') as file:
                self.files = file.read().splitlines()
            self.folder = 'test'
            self.format = ''
        else:
            raise ValueError()

    def __getitem__(self, index):
        img, img_noise = super().__getitem__(index)
        if self.flip:
            if img.shape[1] > img.shape[2]:
                img = img.permute(0, 2, 1)
                img_noise = img_noise.permute(0, 2, 1)
        return img, img_noise

    def __len__(self):
        return len(self.files)

    def image_path(self, index):
        return os.path.join(self.path, 'images', self.folder, self.files[index] + self.format)


class Test12_for_denoising(dataset_for_denoising):
    """
    Load test12 images and add noise
    """
    def __init__(self, pathdir, img_size=[321, 481], clip_to_realistic=True,
                 augmentations=False, normalize=False, noise_std=0.25, resize=False):
        super().__init__(pathdir, 'test', img_size, True, clip_to_realistic,
                         augmentations, normalize, noise_std, resize)

        # Dataset parameters
        self.mean_rgb = [0.43423736, 0.44302556, 0.36703733]
        self.std_rgb = [0.24956392, 0.2343471 , 0.2418366]

        # Get files and dataset folders
        self.files = os.listdir(pathdir)

    def image_path(self, index):
        return os.path.join(self.path, self.files[index])

    def __len__(self):
        return len(self.files)
