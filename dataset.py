import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_val_img, load_mask, load_val_mask, Augment_RGB_torch
import torch.nn.functional as F
import random

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]

        self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index])
        mask = torch.from_numpy(np.float32(mask))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        mask = mask[r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        mask = getattr(augment, apply_trans)(mask)
        mask = torch.unsqueeze(mask, dim=0)
        return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files if is_png_file(x)]


        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        mask = load_mask(self.mask_filenames[tar_index])
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename
