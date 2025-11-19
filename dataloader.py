import functools
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torch.nn.functional as F


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class AGIQADataset_1k(Dataset):
    def __init__(self, csv_file, image_dir, preprocess, test, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file)
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = image_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test

    def __getitem__(self, index):
        image_name = self.data.iloc[index]['Image']
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)  # shape: [3, H, W]

        # 生成224整图与9块裁剪块
        C, H, W = I.shape
        I_resized = F.interpolate(I.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        # 裁切块大小
        patch_size = 384

        stride_h = max((H - patch_size) // 2, 0)
        stride_w = max((W - patch_size) // 2, 0)
        patches = []
        for i in range(3):
            for j in range(3):
                h_start = i * stride_h
                w_start = j * stride_w
                if h_start + patch_size > H:
                    h_start = H - patch_size
                if w_start + patch_size > W:
                    w_start = W - patch_size
                patch = I[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
                patch_resized = F.interpolate(patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                patches.append(patch_resized)
        patches = torch.stack(patches, dim=0)  # (9, 3, 224, 224)

        mos = float(self.data.iloc[index]['MOS'])
        prompt = self.data.iloc[index]['Prompt']

        sample = {'I': I_resized, 'patches': patches, 'prompt': prompt, 'MOS': mos}
        return sample

    def __len__(self):
        return len(self.data.index)


class AGIQADataset_3k(Dataset):
    def __init__(self, csv_file, image_dir, preprocess, test, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file)
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = image_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test

    def __getitem__(self, index):
        image_name = self.data.iloc[index]['name']
        image_path = os.path.join(self.img_dir, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)  # shape: [3, H, W]
        
        # 生成224整图与9块裁剪块
        C, H, W = I.shape
        I_resized = F.interpolate(I.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        # 裁切块大小
        patch_size = 384

        stride_h = max((H - patch_size) // 2, 0)
        stride_w = max((W - patch_size) // 2, 0)
        patches = []
        for i in range(3):
            for j in range(3):
                h_start = i * stride_h
                w_start = j * stride_w
                if h_start + patch_size > H:
                    h_start = H - patch_size
                if w_start + patch_size > W:
                    w_start = W - patch_size
                patch = I[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
                patch_resized = F.interpolate(patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                patches.append(patch_resized)
        patches = torch.stack(patches, dim=0)  # (9, 3, 224, 224)
        
        # 返回所有分数
        mos_quality = float(self.data.iloc[index]['mos_quality'])
        mos_align = float(self.data.iloc[index]['mos_align'])
        prompt = self.data.iloc[index]['prompt']

        sample = {
            'I': I_resized, 
            'patches': patches, 
            'prompt': prompt, 
            'mos_quality': mos_quality,
            'mos_align': mos_align
        }
        return sample

    def __len__(self):
        return len(self.data.index)


class AGIQADataset_2023(Dataset):
    def __init__(self, csv_file, image_dir, preprocess, test, get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file)
        print('%d csv data successfully loaded!' % self.__len__())

        self.img_dir = image_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test

    def __getitem__(self, index):
        image_name = self.data.iloc[index]['image_name']
        model_name = self.data.iloc[index]['model']
        image_path = os.path.join(self.img_dir, model_name, image_name)
        I = self.loader(image_path)
        I = self.preprocess(I)  # shape: [3, H, W]
        
        # 生成224整图与9块裁剪块
        C, H, W = I.shape
        I_resized = F.interpolate(I.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        # 裁切块大小
        patch_size = 384

        stride_h = max((H - patch_size) // 2, 0)
        stride_w = max((W - patch_size) // 2, 0)
        patches = []
        for i in range(3):
            for j in range(3):
                h_start = i * stride_h
                w_start = j * stride_w
                if h_start + patch_size > H:
                    h_start = H - patch_size
                if w_start + patch_size > W:
                    w_start = W - patch_size
                patch = I[:, h_start:h_start+patch_size, w_start:w_start+patch_size]
                patch_resized = F.interpolate(patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                patches.append(patch_resized)
        patches = torch.stack(patches, dim=0)  # (9, 3, 224, 224)
        
        # 返回所有分数
        mosz1 = float(self.data.iloc[index]['mosz1'])
        mosz2 = float(self.data.iloc[index]['mosz2'])
        mosz3 = float(self.data.iloc[index]['mosz3'])
        prompt = self.data.iloc[index]['prompt']

        sample = {
            'I': I_resized, 
            'patches': patches, 
            'prompt': prompt, 
            'mosz1': mosz1,
            'mosz2': mosz2,
            'mosz3': mosz3
        }
        return sample

    def __len__(self):
        return len(self.data.index)

