
import torch

import glob
import scipy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import nltk
nltk.download('wordnet')
import numpy as np


from PIL import Image
from pytorch_pretrained_biggan import one_hot_from_names


imagenet_path_drive = "/content/drive/MyDrive/Colab Notebooks/CV/project/imagenet"
path_drive = "/content/drive/MyDrive/Colab Notebooks/CV/project/"

class INColorDataset(Dataset):
    def __init__(self, subset, fig_size, split, transform=None):

        all_imgs = sorted(glob.glob("/content/imagenet/*"))
        labels = []

        #ground truth labels
        with open(path_drive + "val_gt.txt") as f:
            for line in f:
                labels += [int(line)]

        self.map = self.get_map()

        #can choose to store only a subset of images
        if(subset == "5k"):
            all_imgs = all_imgs[:5000]
            labels = labels[:5000]
        elif(subset == "10k"):
            all_imgs = all_imgs[:10000]
            labels = labels[:10000]

        #picking train, val, test splits given split argument
        all_imgs_size = len(all_imgs)
        if(split == "train"):
            self.imgs = all_imgs[:int(0.6 * all_imgs_size)]
            self.labels = labels[:int(0.6 * all_imgs_size)]
            
        elif(split == "val"):
            self.imgs = all_imgs[int(0.6 * all_imgs_size) : int(0.8 * all_imgs_size)]
            self.labels = labels[int(0.6 * all_imgs_size) : int(0.8 * all_imgs_size)]
        else:
            self.imgs = all_imgs[int(0.8 * all_imgs_size) : ]
            self.labels = labels[int(0.8 * all_imgs_size) : ]
        
        self.transform = transform
        self.fig_size = fig_size

    def __len__(self):
        return len(self.imgs)

    def get_map(self):
        #mapping from ground truth labels in val_gt.txt to standard ImageNet 1000 classes
        metadata = scipy.io.loadmat(path_drive + "meta.mat",struct_as_record=False)
        synsets = np.squeeze(metadata['synsets'])
        map = {s.ILSVRC2012_ID[0][0] : s.words[0].split(",")[0] for s in synsets[:1000]}
        return map

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)

        img = img.convert("RGB")

        #transforms for grayscale and colored versions
        color_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.fig_size, self.fig_size))                                   
        ])

        gray_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.fig_size, self.fig_size)),
                transforms.Grayscale(num_output_channels=3)                            
        ])

        label = self.labels[idx]
        label = one_hot_from_names([self.map[label]],batch_size=1).argmax()
        
        color = color_trans(img)
        gray = gray_trans(img)


        return gray, color, label
    

def get_dataloader(subset, img_size, split, batch_size=32, num_workers=4):
    '''Use Pytorch torch.utils.data.DataLoader to load batched data'''
    
    dataset = INColorDataset(subset, img_size, split)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )
    return dataloader