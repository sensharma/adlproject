import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import shap

import matplotlib.pyplot as plt

from functools import partial
import os

from models.cnn import Net

from data.dataloaders import TORCH_DATAPATH
from paths import MNIST_CNN_PATH as PATH
batch_size = 10000 # 128

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
# device = 'cpu'

# PATH = os.path.abspath('./saved_models/test_exp')

model = Net()
checkpoint = torch.load(f'{PATH}/checkpoint', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(TORCH_DATAPATH, train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=batch_size, shuffle=False)


batch = next(iter(test_loader))
images, _ = batch

background = images[:100]
test_images = images[100:103]

e = shap.DeepExplainer(model, background.to(device))
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

# plot the feature attributions
shap.image_plot(shap_numpy, -test_numpy)

"""
def predict_probs(images, model, h=28, w=28, n_channels=1):
    if type(images) is not torch.Tensor:
        images_tensor = torch.from_numpy(images).to(device)
    else:
        images_tensor = images.to(device)

    if images_tensor.ndim != 4:
        images_tensor.reshape(images_tensor.shape[0], n_channels, h, w)

    with torch.no_grad():
        output = model(images_tensor)
#         return output
        out_probs = output.cpu().numpy().astype(np.float64)
    return out_probs


pred_probs_func = partial(predict_probs, model=model.to(device))

inp = images[:5]
# background = np.zeros_like(inp.numpy())
# test_background = np.ones_like(inp.numpy())
# background = torch.from_numpy(np.zeros_like(inp.numpy()))

ker_background = np.zeros((1, 784))
background = images[:100]
test_images = images[100:103]

plt.imshow(inp[0].squeeze())
plt.imshow(background[0].squeeze())


def kernel_transform(X):
    if type(X) == torch.Tensor:
        X = X.clone()
        X = X.reshape(X.shape[0], X.shape[2] * X.shape[3]).numpy()
    elif type(X) == np.ndarray:
        X = X.copy()
        X = X.reshape(X.shape[0], X.shape[2] * X.shape[3])
    else:
        raise("Unknown datatype passed to kernel_transform")
    return X


# explainer instance created with 0s as background or reference)
# explainer = shap.KernelExplainer(pred_probs_func, background.numpy())
explainer = shap.KernelExplainer(pred_probs_func, ker_background)

deep_exp = shap.DeepExplainer(model, background.to(device))
# dim of X to be entered = #samples x #features
# shap_values = explainer.shap_values(np.ones((5, 784)), nsamples=10)  # runs model 10 times

# X: numpy.array or pandas.DataFrame or any scipy.sparse matrix
# A matrix of samples (# samples x # features) on which to explain
# the model’s output.

# test_im_inp = kernel_transform(test_images)
test_im_inp = kernel_transform(np.ones((1, 1, 28, 28)))

# kernelSHAP returns a list of 10 items (per class), each with dim of
# number of images in test_im_inp and 1 (1 per input image). How to
# get pixel wise, for this?

shap_vals_ker = explainer.shap_values(test_im_inp, nsamples=10)  # runs model 10 times
print(f'shap_vals_ker: {shap_vals_ker}')


shap_vals_deep = deep_exp.shap_values(test_images)

# The above is explaining 5 white image -> np.ones
# To use actual test_images:
# num_test_samples = 5
# test_images = images[:num_test_samples].clone()


# test_images = (test_images.reshape(test_images.shape[0],
#                                   test_images.shape[2]*test_images.shape[3])
#                                   .numpy())


# shap_test_values = explainer.shap_values(test_im_inp, nsamples=10) # runs model 10 times

print(f'shap_vals_deep: {shap_vals_deep}')
"""