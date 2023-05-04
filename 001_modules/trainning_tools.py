# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import optim
from torch.optim import SGD, Adam
from tensorflow.python.client import device_lib
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import matplotlib
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import warnings



## Configs
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
fashion_mnist = datasets.FashionMNIST(
    "../00_data", download=True, train=True
)
tr_images = fashion_mnist.data
tr_targets = fashion_mnist.targets
val_fmnist =datasets.FashionMNIST("../00_data",download=True, \
train=False)
val_images = val_fmnist.data
val_targets = val_fmnist.targets


## Tools
class FMNISTDataset(Dataset):
    def __init__(self, x, y):
        x = x.float() / 255
        x = x.view(-1, 28 * 28)
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)


def get_data(batch_size=32):
    train = FMNISTDataset(tr_images, tr_targets)
    trn_dl = DataLoader(train, batch_size, shuffle=True)
    val = FMNISTDataset(val_images, val_targets)
    val_dl = DataLoader(val, batch_size=len(val_images),
    shuffle=False)
    return trn_dl, val_dl


def get_model(opt=""):
    print(f'using: {device}')
    model = nn.Sequential(
        nn.Linear(28 * 28, 1000),
        nn.ReLU(),
        nn.Linear(1000, 10)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    if opt == "sgd":
        optimizer = SGD(model.parameters(), lr=1e-2)
    else:
        optimizer = Adam(model.parameters(), lr=1e-2)
    
    return model, loss_fn, optimizer


def train_batch(x, y, model, opt, loss_fn):
    model.train() 
    prediction = model(x)
    # compute loss
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    # apply new-weights = f(old-weights, old-weight-gradients)
    # where "f" is the optimizer
    opt.step()
    # Flush gradients memory for next batch of calculations
    opt.zero_grad()
    return batch_loss.item()

def accuracy(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        max_values, argmaxes = prediction.max(-1)
        is_correct = argmaxes == y
        return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()