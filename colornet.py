import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import nltk
nltk.download('wordnet')

from pytorch_pretrained_biggan import BigGAN

class ColorNet(nn.Module):

    """
    Main class defined for ColorNet model
    """

    def __init__(self, out_size = 128):
        """
        z_dim: fixed to 128, dimension of latent embeddings
        out_size: output image size 128x128
        """
        super().__init__()

        self.encoder = self.encoder_setup(name="resnet")
        self.classifier = self.classifier_setup(name="efficient")

        '''
        FC layers mapping from 2048 (output of encoder) to lower dim for BIGGAN
        '''
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

        self.gan = BigGAN.from_pretrained(f'biggan-deep-{out_size}')

    def encoder_setup(self, name="resnet"):

        '''
        pretrained resnet used to encode images to lower dim representation
        '''
        model = None
        if(name == "resnet"):
            model = torchvision.models.resnet50(pretrained=True)
        elif(name == "efficient"):
            model = torchvision.models.efficientnet_b7(pretrained=True)

        model = torch.nn.Sequential(*(list(model.children())[:-1]))

        return model

    def classifier_setup(self, name="resnet"):
        '''
        BIGGAN also takes class prediction so I use
        another resnet to output class probabilities
        '''
        model = None
        if(name == "resnet"):
            model = torchvision.models.resnet50(pretrained=True)
        elif(name == "efficient"):
            model = torchvision.models.efficientnet_b7(pretrained=True)
        return model

    def forward(self, x, gt=None, truncation=1):
        x = transforms.Normalize(mean=[0.445, 0.445, 0.445],
                                 std=[0.269, 0.269, 0.269])(x)

        probs = self.classifier(x)
        labels = None

        #either use ground truth labels or our own class prediction
        if(gt == None):
          ##one hot the max label
          labels = torch.argmax(probs, dim=1)
          labels = F.one_hot(labels, num_classes=1000).type(torch.float32)
        else:
          labels = F.one_hot(gt, num_classes=1000).type(torch.float32)


        encoding = self.encoder(x)
        encoding = nn.Flatten()(encoding)

        encoding = self.fc1(encoding)
        encoding = self.bn1(encoding)
        encoding = F.relu(encoding)
        encoding = self.fc2(encoding)
        encoding = self.bn2(encoding)

        encoding = encoding * truncation
        '''
        gan takes three arguments: encoding, class probs, and truncation factor (set to 1 here for no truncation)
        '''
        output = self.gan(encoding, labels, 1)

        '''
        outputs of BIGGAN images are scaled in [-1, 1], so scale it back to [0,1]
        '''
        norm_output = (output + 1) / 2
        return norm_output, encoding, labels
        


class ColorNetInv(nn.Module):

    '''
    Main class defined for ColorNet model (GAN inversion)
    Start with latent vector directly; no need for encoder
    BIGGAN: https://github.com/huggingface/pytorch-pretrained-BigGAN
    '''

    def __init__(self, out_size = 128):
        super().__init__()

        '''
        FC layers mapping from 2048 (output of encoder) to lower dim for BIGGAN
        '''

        self.gan = BigGAN.from_pretrained(f'biggan-deep-{out_size}')

    def forward(self, x, gt):

        labels = F.one_hot(gt, num_classes=1000).type(torch.float32)
      
        '''
        gan takes three arguments: encoding, class probs, and truncation factor (set to 1 here for no truncation)
        '''
        output = self.gan(x, labels, 1)

        '''
        outputs of BIGGAN images are scaled in [-1, 1], so scale it back to [0,1]
        '''
        norm_output = (output + 1) / 2
        return norm_output
        