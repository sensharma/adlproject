import torch
from torchvision import datasets, transforms
import torchtext as tt
# from torch.utils.data import Subset

import os

# TORCH_DATAPATH = os.path.join(os.path.dirname(__file__)) 
TORCH_DATAPATH = os.getcwd() 

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

# create the dataloader for the MNIST
# MNIST train dataloader


def mnist_loader(train=True,
                 path=TORCH_DATAPATH,
                 train_limit=False,
                 test_limit=False,
                 batch_size=32,
                 shuffle=False,
                 randomtestsample=False,
                 rand_indices=False):

    """[MNIST train/test dataloader]

    Args:
        train (bool, optional): [False returns test data.]. Defaults to True.
        path ([str], optional): [path on machine for data]. Defaults to DATAPATH. 
        train_limit (bool, optional): [e.g. if 500, returns first 500 data samples]. Defaults to False. False = Full dataset.
        test_limit (bool, optional): [Same as train_limit, for test data]. Defaults to False.
        batch_size (int, optional): [mini-batch size]. Defaults to 32.

    Returns:
        [pytorch dataloader]: [train/test dataloader]
    """

    # creating train and test datasets
    train_dataset = datasets.MNIST(root=path,
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor()
                                   ])
                                   )

    test_dataset = datasets.MNIST(root=path,
                                  train=False,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ])
                                  )

    # if randomsample:
    #    test_indices = np.arange(len(X_test))
    #    rand_indices = np.random.choice(test_indices, size=batch_size, replace=False)
    #    X_test_rand = X_test[rand_indices, :, :, :]
    #    y_test_rand = y_test[rand_indices, :, :, :]
    #    return X_test_rand, y_test_rand, rand_indices

    if randomtestsample:
        test_dataset = torch.utils.data.Subset(
            test_dataset, rand_indices)

    # creating data subsets if limits not False
    if train_limit:
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(0, train_limit))
    if test_limit:
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(0, test_limit))

    # creating and returning data_loader, depending on options provided
    if train:
        mnist_train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return mnist_train_loader
    else:
        # MNIST test dataloader
        mnist_test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return mnist_test_loader


def imdb_loader(path=TORCH_DATAPATH,
                batch_size=4
                ):

    train_iter, test_iter = tt.datasets.IMDB.iters(batch_size=batch_size,
                                                   root=path,
                                                   device=device)
    return train_iter, test_iter
